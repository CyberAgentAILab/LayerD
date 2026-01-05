# Contributing to LayerD

Thank you for your interest in contributing to LayerD! This document provides guidelines and instructions for contributing.

## Ways to Contribute

We welcome contributions in many forms:

- **Bug reports**: Report issues you encounter
- **Feature requests**: Suggest new features or improvements
- **Code contributions**: Submit pull requests with bug fixes or new features
- **Documentation**: Improve or expand documentation
- **Examples**: Add usage examples or tutorials
- **Testing**: Help test new features or report edge cases

## Getting Started

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/LayerD.git
   cd LayerD
   ```

3. **Install dependencies** with uv:

   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install all dependencies including dev tools
   uv sync --all-extras
   ```

4. **Verify installation**:

   ```bash
   # Run tests
   uv run pytest

   # Run type checking
   uv run mypy src/ tests/ \
     vendor/simple-lama-inpainting/simple_lama_inpainting/ \
     vendor/cr-renderer/src/cr_renderer/

   # Run linting
   uv run ruff check src/ tests/ vendor/
   ```

For detailed development setup, see [docs/development.md](docs/development.md).

## Development Workflow

### Creating a Feature Branch

1. Create a branch from main:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. Use descriptive branch names:
   - `feature/add-custom-model-support`
   - `fix/cuda-memory-leak`
   - `docs/improve-training-guide`

### Making Changes

1. **Write code** following our style guidelines (see below)

2. **Add tests** for new features or bug fixes:

   ```python
   # tests/test_your_feature.py
   def test_your_feature():
       # Your test code
       assert expected == actual
   ```

3. **Update documentation** if needed:
   - Update relevant files in `docs/`
   - Add docstrings to new functions/classes
   - Update CLAUDE.md if changing architecture

4. **Run quality checks**:

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

### Committing Changes

We follow conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

Example commit messages:

```bash
git commit -m "feat: add support for custom matting models"
git commit -m "fix: resolve CUDA memory leak in training loop"
git commit -m "docs: improve installation instructions"
```

For larger changes, include a description:

```bash
git commit -m "feat: add support for custom matting models

- Add model registry pattern
- Update LayerD class to accept model name
- Add tests for custom model loading
- Update documentation"
```

## Code Style Guidelines

### Type Annotations

LayerD uses **strict mypy type checking**. All functions must have complete type annotations:

```python
# Bad - no type annotations
def process_image(image):
    return decompose(image)

# Good - complete type annotations
def process_image(image: Image.Image) -> list[Image.Image]:
    return decompose(image)
```

Type checking rules:

- `disallow_untyped_defs=true`
- `disallow_incomplete_defs=true`
- `no_implicit_optional=true`

### Code Formatting

We use **Ruff** for linting and formatting:

```bash
# Format code
uv run ruff format src/

# Check for linting issues
uv run ruff check src/

# Auto-fix linting issues
uv run ruff check src/ --fix
```

### Docstrings

Use clear docstrings for public functions and classes:

```python
def decompose(
    self,
    image: Image.Image,
    max_iterations: int = 3
) -> list[Image.Image]:
    """Decompose an image into layers.

    Args:
        image: Input image in RGB or RGBA format
        max_iterations: Maximum number of decomposition iterations

    Returns:
        List of PIL Images in RGBA format, ordered as
        [background, topmost_fg, ..., bottommost_fg]
    """
    ...
```

### Import Organization

Group imports in this order:

1. Standard library
2. Third-party packages
3. Local imports

```python
# Standard library
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
from PIL import Image

# Local
from layerd.models import LayerD
```

## Testing Guidelines

### Writing Tests

1. **Test file naming**: `test_*.py` in the `tests/` directory

2. **Test function naming**: Use descriptive names

   ```python
   # Bad
   def test_1():
       ...

   # Good
   def test_decompose_returns_correct_number_of_layers():
       ...
   ```

3. **Use fixtures** for common setup (see `tests/conftest.py`)

4. **Test edge cases**:
   - Empty inputs
   - Invalid inputs
   - Boundary conditions
   - Error handling

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_basic_decompose.py

# Run specific test
uv run pytest tests/test_basic_decompose.py::test_decompose

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=layerd
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:

   ```bash
   uv run pytest
   ```

2. **Run type checking**:

   ```bash
   uv run mypy src/ tests/ \
     vendor/simple-lama-inpainting/simple_lama_inpainting/ \
     vendor/cr-renderer/src/cr_renderer/
   ```

3. **Format and lint code**:

   ```bash
   uv run ruff format src/ tests/ vendor/
   uv run ruff check src/ tests/ vendor/
   ```

4. **Update documentation** if needed

5. **Commit your changes** with descriptive messages

### Submitting a Pull Request

1. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a pull request** on GitHub:
   - Go to the original LayerD repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR template

3. **PR description should include**:
   - Summary of changes
   - Motivation and context
   - Related issue numbers (if applicable)
   - Screenshots (for UI/visual changes)
   - Testing done

### PR Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, maintainers will merge your PR

Tips:

- Respond to review comments promptly
- Keep PRs focused (one feature/fix per PR)
- Be open to feedback and suggestions

## Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **LayerD version**: `pip show layerd`
2. **Python version**: `python --version`
3. **Operating system**: Linux, macOS, Windows
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Steps to reproduce**: Minimal example to reproduce the bug
7. **Error messages**: Full traceback if available

Example bug report:

```markdown
**Environment:**
- LayerD version: 0.1.0
- Python version: 3.12.3
- OS: Ubuntu 22.04

**Description:**
LayerD crashes when processing images larger than 4096x4096

**Steps to reproduce:**
1. Load a 5000x5000 PNG image
2. Run decompose()
3. Get CUDA out of memory error

**Expected:** Should process or give helpful error message
**Actual:** Crashes with CUDA OOM

**Error message:**
```

RuntimeError: CUDA out of memory...

```
```

### Feature Requests

When requesting features, include:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches you've considered
4. **Additional context**: Examples, references, etc.

## Code Review Guidelines

When reviewing pull requests:

- Be respectful and constructive
- Focus on the code, not the person
- Explain reasoning behind suggestions
- Recognize good work
- Suggest alternatives when requesting changes

## Community Guidelines

- Be respectful and inclusive
- Welcome newcomers
- Help others learn
- Give credit where due
- Assume good intentions

## Questions?

If you have questions about contributing:

- Check the documentation in [docs/](docs/)
- Look for existing issues on GitHub
- Create a new issue with your question
- Reach out to maintainers

## License

By contributing to LayerD, you agree that your contributions will be licensed under the Apache-2.0 License.

## Acknowledgments

Thank you for contributing to LayerD! Your efforts help make this project better for everyone.

## Related Documentation

- [Development Guide](docs/development.md) - Detailed development setup and workflows
- [Architecture](docs/architecture.md) - Understanding the codebase
- [Testing Guide](docs/development.md#testing) - Testing best practices
