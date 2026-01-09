# Claude Code Skills for LayerD

This directory contains custom skills (slash commands) for LayerD development.

## Available Skills

### `/check` - Run quality checks

Run code quality checks including type checking, linting, formatting, and tests.

```bash
/check           # Run all checks (default)
/check mypy      # Run type checking only
/check ruff      # Run linting only
/check format    # Check code formatting only
/check test      # Run tests only
```

**What it does:**

- **mypy**: Type checks `src/`, `tests/`, and vendored packages
- **ruff**: Lints code in `src/`, `tests/`, and `vendor/`
- **format**: Checks code formatting (no changes made)
- **test**: Runs the full test suite with pytest

### `/fix` - Auto-fix code issues

Automatically fix linting issues and format code.

```bash
/fix             # Run all auto-fixes (default)
/fix ruff        # Auto-fix linting issues only
/fix format      # Auto-format code only
```

**What it does:**

- **ruff**: Automatically fixes linting issues where possible
- **format**: Formats code according to project style

## Usage in Claude Code

These skills can be invoked directly in the Claude Code chat:

```
/check          # Run all quality checks
/fix            # Auto-fix and format code
/check mypy     # Just run type checking
```

## Implementation

Skills are implemented as Bash scripts in this directory. Each skill must:

1. Be executable (`chmod +x`)
2. Have a `.sh` extension
3. Handle command-line arguments
4. Provide helpful usage information

## Adding New Skills

To add a new skill:

1. Create a new `.sh` file in this directory
2. Make it executable: `chmod +x your-skill.sh`
3. Document it in this README
4. Test it: `/your-skill`

## Notes

- Skills run in the project root directory
- Use `uv run` prefix for Python tools to ensure correct environment
- The `.claude/` directory is gitignored (local configuration only)
