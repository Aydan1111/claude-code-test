# CLAUDE.md - AI Assistant Guide

This file provides guidance for AI assistants (like Claude) working on this codebase.

## Project Overview

**Repository:** claude-code-test
**Owner:** Aydan1111
**Status:** New/Initial Setup

This repository is in its early stages of development. As the project evolves, this document should be updated to reflect the current architecture, conventions, and workflows.

## Repository Structure

```
claude-code-test/
├── README.md          # Project description and documentation
├── CLAUDE.md          # This file - AI assistant guidance
└── .git/              # Git version control
```

## Development Workflow

### Branch Naming Convention

- Feature branches: `feature/<description>`
- Bug fixes: `fix/<description>`
- AI/Claude branches: `claude/<description>-<session-id>`

### Commit Guidelines

1. Write clear, descriptive commit messages
2. Use conventional commit format when applicable:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code refactoring
   - `test:` for adding/updating tests
   - `chore:` for maintenance tasks

3. Keep commits atomic and focused on a single change

### Git Operations

- Always push with tracking: `git push -u origin <branch-name>`
- Fetch specific branches when possible: `git fetch origin <branch-name>`
- Pull with origin specified: `git pull origin <branch-name>`

## Code Conventions

### General Principles

1. **Simplicity First:** Keep solutions simple and focused on the task at hand
2. **Avoid Over-Engineering:** Don't add features beyond what's requested
3. **Security Awareness:** Be mindful of OWASP Top 10 vulnerabilities
4. **Clean Code:** Write readable, maintainable code

### File Organization

As the project grows, organize files logically:
- Source code in `src/` directory
- Tests in `tests/` or alongside source files
- Configuration files in project root
- Documentation in `docs/` or project root

## For AI Assistants

### Before Making Changes

1. Read and understand existing code before modifying
2. Check for existing patterns and conventions in the codebase
3. Verify the current git status and branch

### When Implementing Features

1. Plan the task using available tools
2. Make minimal, focused changes
3. Follow existing code style and patterns
4. Test changes when testing infrastructure exists

### When Fixing Bugs

1. Understand the root cause before implementing a fix
2. Avoid introducing new bugs or security vulnerabilities
3. Keep fixes targeted - don't refactor unrelated code

### Communication

1. Provide clear explanations of changes made
2. Reference specific file paths and line numbers when discussing code
3. Ask clarifying questions when requirements are ambiguous

## Common Commands

```bash
# Check repository status
git status

# View recent commits
git log --oneline -10

# Create and switch to a new branch
git checkout -b <branch-name>

# Stage and commit changes
git add <files>
git commit -m "type: description"

# Push changes
git push -u origin <branch-name>
```

## Testing

*Testing infrastructure to be added as the project develops.*

## Build & Deployment

*Build and deployment processes to be documented as they are established.*

---

**Last Updated:** 2026-02-02
**Maintained By:** AI assistants and project contributors
