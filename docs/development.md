# Development Guide

This guide covers development setup, testing, and contributing workflows for LayerD.

## Development Environment Setup

LayerD uses [uv](https://docs.astral.sh/uv/) to manage the development environment.

### Installation Options

```bash
# Install core dependencies + dev tools (default behavior)
uv sync

# Install core dependencies only (no dev tools, no optional dependencies)
uv sync --no-default-groups

# Install with all optional dependencies + dev tools (recommended for contributors)
uv sync --all-extras

# Install with specific optional dependencies + dev tools
uv sync --extra dataset
uv sync --extra train
```

### Dependencies Overview

- **Core dependencies**: Required for basic inference (matting, inpainting)
- **`dataset` extra**: Required for Crello dataset generation - installed with `--extra dataset`
- **`train` extra**: Required for training the matting module - installed with `--extra train`
- **`dev` group**: Development tools (pytest, mypy, ruff, pre-commit) - **included by default**

**Note**: `uv sync` installs core dependencies and dev tools by default. Use `uv sync --no-default-groups` to exclude dev tools.

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with image output saved
uv run pytest --save-images

# Run tests with custom matting process size
uv run pytest --matting-process-size 512 512

# Run specific test
uv run pytest tests/test_basic_decompose.py::test_decompose

# Run with verbose output
uv run pytest -v
```

### Test Configuration

- **Test fixtures**: Defined in `tests/conftest.py`
- **Custom options**:
  - `--save-images`: Save test outputs to `tests/output/` directory
  - `--matting-process-size`: Override default matting model process size
- **Test outputs**: Saved to `tests/output/` (gitignored)
- **Warning filters**: FutureWarnings from timm library are filtered (configured in `pyproject.toml`)

### Writing Tests

When writing tests:

1. Use pytest fixtures for common setup
2. Add type annotations to all test functions
3. Use descriptive test names that explain what is being tested
4. Clean up any temporary files created during tests

## Code Quality

### Type Checking

LayerD uses strict mypy configuration:

```bash
# Run type checking
uv run mypy src/ tests/ \
  vendor/simple-lama-inpainting/simple_lama_inpainting/ \
  vendor/cr-renderer/src/cr_renderer/
```

**Type checking rules:**

- `disallow_untyped_defs=true` - All functions must have type annotations
- `disallow_incomplete_defs=true` - All parameters and return types must be annotated
- `no_implicit_optional=true` - Optional types must be explicit

All functions must have complete type annotations. This helps catch bugs early and provides better IDE support.

**Note**: `src/layerd/_vendor/` is excluded from type checking (configured in pyproject.toml). We check the source packages in `vendor/` but not the bundled copies in `_vendor/`.

### Linting

```bash
# Run linting
uv run ruff check src/ tests/ vendor/

# Auto-fix linting issues
uv run ruff check src/ tests/ vendor/ --fix
```

### Code Formatting

```bash
# Check code formatting
uv run ruff format src/ tests/ vendor/ --check

# Format code
uv run ruff format src/ tests/ vendor/
```

### Pre-commit Checks

Before committing code, run:

```bash
# Type checking
uv run mypy src/ tests/ \
  vendor/simple-lama-inpainting/simple_lama_inpainting/ \
  vendor/cr-renderer/src/cr_renderer/

# Linting
uv run ruff check src/ tests/ vendor/

# Formatting
uv run ruff format src/ tests/ vendor/

# Tests
uv run pytest
```

**Claude Code users**: Use the custom skills for convenience:

```bash
/check         # Run all quality checks
/check mypy    # Just type checking
/fix           # Auto-fix linting and format code
```

See [.claude/skills/README.md](../.claude/skills/README.md) for more details.

## Vendored Dependencies

LayerD bundles two dependencies under `layerd._vendor` to enable numpy 2.0 compatibility. These dependencies are maintained using git subtree.

### Directory Structure

- **`vendor/`**: Source of truth for git subtree operations (tracked in git)
- **`src/layerd/_vendor/`**: Bundled copy for distribution (tracked in git)

Both directories are committed to git to ensure `pip install git+...` and editable installs work correctly.

### Syncing Vendored Dependencies

When updating vendored dependencies from upstream:

#### 1. Pull Updates Using Git Subtree

```bash
# Update simple-lama-inpainting
git subtree pull --prefix vendor/simple-lama-inpainting \
  https://github.com/enesmsahin/simple-lama-inpainting.git main --squash

# Update cr-renderer
git subtree pull --prefix vendor/cr-renderer \
  https://github.com/CyberAgentAILab/cr-renderer.git main --squash
```

#### 2. Sync to Bundled Copy

##### Option A: Automatic (if git hook installed)

The post-merge hook will automatically sync changes. Review and stage them:

```bash
git status
git add src/layerd/_vendor/
```

##### Option B: Manual Sync

```bash
# Use the sync script
./tools/sync-vendor.sh

# Or manually:
rm -rf src/layerd/_vendor/simple_lama_inpainting
cp -r vendor/simple-lama-inpainting/simple_lama_inpainting/ \
  src/layerd/_vendor/simple_lama_inpainting/

rm -rf src/layerd/_vendor/cr_renderer
cp -r vendor/cr-renderer/cr_renderer/ \
  src/layerd/_vendor/cr_renderer/
```

#### 3. Test the Changes

```bash
uv run pytest
uv run mypy src/
```

#### 4. Commit Both Directories

```bash
git add vendor/ src/layerd/_vendor/
git commit -m "chore: update vendored dependencies from upstream"
```

### Git Hook Setup (Optional but Recommended)

To automatically sync `vendor/` → `_vendor` after git operations:

```bash
# Copy the post-merge hook
cp tools/post-merge.sample .git/hooks/post-merge
chmod +x .git/hooks/post-merge
```

## Git Workflow

### Branch Strategy

1. **main**: Stable branch with released code
2. **feature branches**: Create from main for new features
3. **bugfix branches**: Create from main for bug fixes

### Commit Messages

Follow conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

Example:

```
feat: add support for custom matting models

- Add model registry pattern
- Update LayerD class to accept model name
- Add tests for custom model loading
```

### Pull Request Process

1. Fork and clone the repository
2. Create a feature branch from main
3. Make your changes
4. Run pre-commit checks (tests, type checking, linting)
5. Commit with descriptive messages
6. Push to your fork
7. Create a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

## Development Tools

### uv Commands

```bash
# Add a new dependency
uv add <package>

# Add a development dependency
uv add --group dev <package>

# Update dependencies
uv lock --upgrade

# Show installed packages
uv pip list
```

### Debugging

For debugging during development:

```bash
# Run with Python debugger
uv run python -m pdb ./tools/infer.py --input image.png --output-dir outputs/

# Run with verbose logging
uv run layerd --input image.png --output-dir outputs/ --log-level DEBUG
```

## Project Structure

Understanding the project structure helps with navigation:

```
LayerD/
├── src/layerd/           # Main source code
│   ├── models/           # Model implementations
│   ├── matting/          # Matting training code
│   ├── data/             # Dataset utilities
│   ├── evaluation/       # Evaluation metrics
│   └── _vendor/          # Bundled dependencies
├── tests/                # Test suite
├── tools/                # Utility scripts
│   ├── infer.py          # Batch inference
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── generate_crello_matting.py  # Dataset generation
├── docs/                 # Documentation
├── vendor/               # Vendored source (git subtree)
└── data/                 # Test images
```

## Related Documentation

- [Architecture](architecture.md) - Code architecture and design patterns
- [Testing Best Practices](testing.md) - Detailed testing guidelines (if available)
- [Contributing Guidelines](../CONTRIBUTING.md) - How to contribute to LayerD
