# ===============================================================
#   🐎🔥 SMART RACING AI (No Odds, No Historical Data Needed) 🔥🐎
# ===============================================================
# Uses:
#   - Pace model
#   - Ground suitability (bucketed)
#   - Distance suitability
#   - Class movement
#   - Trainer / jockey as DRIVERS (strong importance + combo score)
#   - Weight & age profiling
#   - Fitness (days since run)
#   - Ensemble weighting (ML-like)
#   - Simple structural filters + anti-hype proxy
#   - Beautiful graphical race output
# ===============================================================

from typing import List, Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# 🐴 Horse Data Structure
# ---------------------------------------------------------------
@dataclass
class Horse:
    name: str
    age: int
    weight: float
    form: str
    ground: str                 # current race ground ("Soft", "Good", "Yielding", etc.)
    distance: float             # race distance (e.g. 2.0 miles)
    last_distances: List[float] # previous race distances
    last_grounds: List[str]     # previous race grounds
    pace_style: str             # "Front", "Prominent", "Mid", "Hold"
    days_since_run: int
    trainer: str
    jockey: str
    race_class: int
    last_race_class: int


# ---------------------------------------------------------------
# 🎯 Smart Feature Scorers (0–10 Domain Values)
# ---------------------------------------------------------------
class FeatureScorer:
    ELITE_TRAINERS = ["Mullins", "Elliott", "O'Brien", "De Bromhead"]
    ELITE_JOCKEYS  = ["Townend", "Blackmore", "Kennedy", "Russell"]

    @staticmethod
    def _is_elite_trainer(trainer: str) -> bool:
        t = (trainer or "").lower()
        return any(x.lower() in t for x in FeatureScorer.ELITE_TRAINERS)

    @staticmethod
    def _is_elite_jockey(jockey: str) -> bool:
        j = (jockey or "").lower()
        return any(x.lower() in j for x in FeatureScorer.ELITE_JOCKEYS)

    @staticmethod
    def score_form(form: str) -> float:
        """
        Scores last ~4 runs. Handles digits and common letters like:
        P/PU (pulled up), F (fell), U/UR (unseated), R (refused), B (brought down)
        """
        if not form or form.strip() in {"-", ""}:
            return 4.0

        # Normalize and strip common separators (dashes, slashes, whitespace)
        s = form.strip().lower()
        s = ''.join(c for c in s if c.isalnum())

        # Weight map: digits good (1 best), letters mostly bad
        weight_map = {
            "1": 10, "2": 7, "3": 5, "4": 3, "5": 2,
            "6": 1, "7": 1, "8": 1, "9": 1, "0": 1,
            "p": 1,  # includes PU often represented as P in some feeds
            "u": 1,
            "f": 1,
            "r": 1,
            "b": 2,
        }
        decay = [0.50, 0.25, 0.15, 0.10]

        score = 0.0
        for i, ch in enumerate(s[:4]):
            score += weight_map.get(ch, 1) * decay[i]
        return float(score)

    @staticmethod
    def score_trainer(trainer: str) -> float:
        return 10.0 if FeatureScorer._is_elite_trainer(trainer) else 6.0

    @staticmethod
    def score_jockey(jockey: str) -> float:
        return 10.0 if FeatureScorer._is_elite_jockey(jockey) else 6.0

    @staticmethod
    def score_combo(trainer: str, jockey: str) -> float:
        """
        Combo score is a REAL driver: elite+elite = strong boost.
        """
        t_elite = FeatureScorer._is_elite_trainer(trainer)
        j_elite = FeatureScorer._is_elite_jockey(jockey)

        if t_elite and j_elite:
            return 10.0
        if t_elite or j_elite:
            return 7.5
        return 5.0

    @staticmethod
    def score_fitness(days: int) -> float:
        if days <= 14: return 10.0
        if days <= 30: return 8.0
        if days <= 45: return 6.0
        return 4.0

    @staticmethod
    def score_weight(weight: float) -> float:
        if weight <= 10: return 10.0
        if weight <= 11: return 8.0
        return 6.0

    @staticmethod
    def score_age(age: int) -> float:
        if 3 <= age <= 5: return 10.0
        if age == 6: return 8.0
        return 6.0

    @staticmethod
    def ground_bucket(g: str) -> str:
        """
        Buckets going strings into a few stable classes so
        "Good to Soft" ~ "Soft" and "Yielding" ~ "Good".
        """
        s = (g or "").strip().lower()
        if "heavy" in s: return "soft"
        if "soft" in s: return "soft"
        if "yield" in s: return "good"
        if "good" in s: return "good"
        if "firm" in s: return "firm"
        return "unknown"

    @staticmethod
    def score_ground(current: str, history: List[str]) -> float:
        if not history:
            return 5.0
        cur_b = FeatureScorer.ground_bucket(current)
        hist_b = [FeatureScorer.ground_bucket(x) for x in history if x]
        matches = sum(1 for b in hist_b if b == cur_b and b != "unknown")
        # base 4 + 2 per match, capped at 10
        return float(min(10.0, 4.0 + matches * 2.0))

    @staticmethod
    def score_distance(current: float, past: List[float]) -> float:
        if not past:
            return 5.0
        diffs = [abs(current - d) for d in past if d is not None]
        if not diffs:
            return 5.0
        avg = sum(diffs) / len(diffs)
        if avg <= 0.5: return 10.0
        if avg <= 1.0: return 8.0
        return 6.0

    @staticmethod
    def score_class_drop(current: int, last: int) -> float:
        if last - current >= 2: return 10.0
        if last - current == 1: return 8.0
        if current == last: return 6.0
        return 4.0

    @staticmethod
    def score_pace(pace: str, field_pace: Dict[str, int]) -> float:
        # Favour lone front-runner; avoid pace wars; favour closers if lots of speed
        front_n = field_pace.get("Front", 0)

        if pace == "Front" and front_n == 1:
            return 10.0
        if pace == "Front" and front_n >= 3:
            return 5.0  # pace collapse risk
        if pace == "Hold" and front_n >= 3:
            return 9.0  # strong closer advantage

        mapping = {"Front": 8.0, "Prominent": 7.0, "Mid": 6.0, "Hold": 5.0}
        return float(mapping.get(pace, 6.0))


# ---------------------------------------------------------------
# 🤖 SMART RACING AI ENGINE
# ---------------------------------------------------------------
class SmartRacingAI:
    """
    Trainer/Jockey as DRIVERS:
      - We include trainer, jockey, and combo_score as features.
      - We apply feature-importance multipliers so these matter more
        even with ensemble randomness.
    """

    def __init__(self, ensembles: int = 200, seed: int = 42):
        self.ensembles = ensembles
        np.random.seed(seed)

        # Feature importance multipliers (drivers)
        self.feature_importance = {
            "form_score": 1.05,
            "ground_score": 1.10,
            "distance_score": 1.10,
            "pace_score": 1.05,
            "class_score": 1.05,
            "fitness": 1.00,
            "weight_score": 0.95,
            "age_score": 0.95,

            # DRIVERS:
            "trainer": 1.25,
            "jockey": 1.25,
            "combo_score": 1.20,
        }

    def analyze_race(self, horses: List[Horse]) -> pd.DataFrame:
        if not horses:
            return pd.DataFrame()

        # Build pace map (safe)
        pace_map = {"Front": 0, "Prominent": 0, "Mid": 0, "Hold": 0}
        for h in horses:
            pace_map[h.pace_style] = pace_map.get(h.pace_style, 0) + 1

        # Build feature table
        rows = []
        for h in horses:
            rows.append({
                "name": h.name,

                "form_score": FeatureScorer.score_form(h.form),

                # DRIVERS
                "trainer": FeatureScorer.score_trainer(h.trainer),
                "jockey": FeatureScorer.score_jockey(h.jockey),
                "combo_score": FeatureScorer.score_combo(h.trainer, h.jockey),

                "fitness": FeatureScorer.score_fitness(h.days_since_run),
                "weight_score": FeatureScorer.score_weight(h.weight),
                "age_score": FeatureScorer.score_age(h.age),

                "ground_score": FeatureScorer.score_ground(h.ground, h.last_grounds),
                "distance_score": FeatureScorer.score_distance(h.distance, h.last_distances),
                "class_score": FeatureScorer.score_class_drop(h.race_class, h.last_race_class),
                "pace_score": FeatureScorer.score_pace(h.pace_style, pace_map),
            })

        df = pd.DataFrame(rows)

        # -------------------------------------------------------
        # Simple base race-fit penalty (structure-first filter)
        # If BOTH ground and distance look poor, cap ceiling.
        # -------------------------------------------------------
        df["profile_penalty"] = 0.0
        df.loc[(df["ground_score"] <= 5.0) & (df["distance_score"] <= 6.0), "profile_penalty"] = -8.0

        # -------------------------------------------------------
        # Ensemble weighting (simulated ML)
        # -------------------------------------------------------
        features = [c for c in df.columns if c not in {"name"}]

        # Build importance array aligned to features
        importance = np.array([self.feature_importance.get(f, 1.0) for f in features], dtype=float)

        final_scores = np.zeros(len(df), dtype=float)

        for _ in range(self.ensembles):
            # Random weights + importance -> makes some features drivers reliably
            rand_w = np.random.uniform(0.5, 1.5, len(features))
            weights = rand_w * importance
            scores = (df[features].values * weights).sum(axis=1)
            final_scores += scores

        df["ai_score"] = (final_scores / self.ensembles)

        # -------------------------------------------------------
        # Anti-hype proxy: don't let reputation alone rescue bad structure
        # (still keeps trainer/jockey as drivers overall)
        # -------------------------------------------------------
        structure = (
            df["ground_score"] +
            df["distance_score"] +
            df["pace_score"] +
            df["class_score"] +
            df["form_score"]
        )
        star_power = df["trainer"] + df["jockey"] + df["combo_score"]

        df["ai_score"] -= np.where((star_power >= 26.0) & (structure <= 28.0), 4.0, 0.0)

        # -------------------------------------------------------
        # Confidence (safe scaling) - used for ranking
        # -------------------------------------------------------
        score_range = float(df["ai_score"].max() - df["ai_score"].min())
        if score_range == 0:
            df["confidence"] = 50.0
        else:
            df["confidence"] = 100.0 * (df["ai_score"] - df["ai_score"].min()) / score_range

        # -------------------------------------------------------
        # Model Confidence (70-90 scale)
        # -------------------------------------------------------
        # IMPORTANT: This is NOT a probability of winning.
        # It represents how well the horse's profile aligns with
        # the model's preferred signals (elite connections, ground
        # fit, distance fit, no structural penalties).
        # Higher = more factors align. Does NOT mean "more likely to win".
        # -------------------------------------------------------
        model_conf = np.full(len(df), 70.0)  # Base confidence = 70

        # Increase for aligned signals (existing scores, no new logic)
        model_conf += np.where(df["trainer"] == 10.0, 5.0, 0.0)       # Elite trainer
        model_conf += np.where(df["jockey"] == 10.0, 5.0, 0.0)        # Elite jockey
        model_conf += np.where(df["ground_score"] >= 8.0, 4.0, 0.0)   # Ground fit
        model_conf += np.where(df["distance_score"] >= 8.0, 4.0, 0.0) # Distance fit
        model_conf += np.where(df["profile_penalty"] == 0.0, 2.0, 0.0) # No penalties

        # Clamp to 70-90 range (never 100%, never below 70%)
        df["model_confidence"] = np.clip(model_conf, 70.0, 90.0)

        return df.sort_values("confidence", ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------
    # 🎨 BEAUTIFUL OUTPUT
    # -----------------------------------------------------------
    def print_results(self, df: pd.DataFrame):
        if df is None or df.empty:
            print("No results to display.")
            return

        print("\n" + "=" * 70)
        print("🐎🔥             **AI SMART RACE PREDICTIONS**              🔥🐎")
        print("=" * 70)

        medals = ["🥇 GOLD", "🥈 SILVER", "🥉 BRONZE"]

        for i, row in df.head(3).iterrows():
            print(f"\n⭐🐴 " + "=" * 60 + " 🏇⭐")
            # Note: model_confidence = alignment with model signals, NOT win probability
            print(f"     {medals[i]} PICK: {str(row['name']).upper()} — {row['model_confidence']:.0f}% Model Alignment")
            print(f"     🏇 Strengths:")

            factors = []
            if row.get("form_score", 0) >= 7: factors.append("Strong recent form")
            if row.get("ground_score", 0) >= 8: factors.append("Ground specialist")
            if row.get("distance_score", 0) >= 8: factors.append("Perfect distance fit")
            if row.get("class_score", 0) >= 8: factors.append("Class advantage")
            if row.get("pace_score", 0) >= 9: factors.append("Strong pace setup")

            # Drivers:
            if row.get("trainer", 0) >= 9: factors.append("Top trainer")
            if row.get("jockey", 0) >= 9: factors.append("Elite jockey")
            if row.get("combo_score", 0) >= 9: factors.append("Elite trainer-jockey combo")

            print("       • " + "\n       • ".join(factors) if factors else "       • Balanced, reliable profile")
            print("⭐🐴 " + "=" * 60 + " 🏇⭐")

        print("\n🌟 This AI uses:")
        print("   • Pace intelligence")
        print("   • Ground & distance simulation")
        print("   • Class movement analysis")
        print("   • Trainer/jockey as key drivers + combo strength")
        print("   • Ensemble-weight ML logic")

        # Clarify what model confidence means
        print("\n⚠️  Model Alignment % = how well factors align, NOT chance to win.")

        print("=" * 70 + "\n")
