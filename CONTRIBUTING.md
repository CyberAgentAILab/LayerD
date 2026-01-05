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
   uv run pytest  # Run tests to verify setup
   ```

For detailed development environment options and tools, see [docs/development.md](docs/development.md).

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

4. **Run quality checks** before committing:

   ```bash
   uv run pytest                              # Run tests
   uv run mypy src/ tests/ vendor/...         # Type checking
   uv run ruff check src/ tests/ vendor/      # Linting
   uv run ruff format src/ tests/ vendor/     # Formatting
   ```

   See [docs/development.md#code-quality](docs/development.md#code-quality) for complete commands and options.

### Committing Changes

We follow conventional commit format:

```bash
git commit -m "feat: add support for custom matting models"
git commit -m "fix: resolve CUDA memory leak in training loop"
git commit -m "docs: improve installation instructions"
```

**Commit types:**

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

## Code Style Guidelines

LayerD follows strict code quality standards:

### Type Annotations

All functions must have complete type annotations:

```python
# Good - complete type annotations
def process_image(image: Image.Image) -> list[Image.Image]:
    return decompose(image)
```

**Requirements:**

- `disallow_untyped_defs=true` - All functions must have type annotations
- `disallow_incomplete_defs=true` - All parameters and return types must be annotated
- `no_implicit_optional=true` - Optional types must be explicit

### Code Formatting

We use **Ruff** for linting and formatting. Run `uv run ruff format src/` before committing.

### Docstrings

Add clear docstrings to public functions and classes:

```python
def decompose(self, image: Image.Image, max_iterations: int = 3) -> list[Image.Image]:
    """Decompose an image into layers.

    Args:
        image: Input image in RGB or RGBA format
        max_iterations: Maximum number of decomposition iterations

    Returns:
        List of PIL Images in RGBA format
    """
```

For complete code style details, see [docs/development.md#code-quality](docs/development.md#code-quality).

## Testing Guidelines

### Writing Tests

1. **Test file naming**: `test_*.py` in the `tests/` directory
2. **Test function naming**: Use descriptive names (e.g., `test_decompose_returns_correct_number_of_layers`)
3. **Use fixtures** for common setup (see `tests/conftest.py`)
4. **Test edge cases**: Empty inputs, invalid inputs, boundary conditions, error handling

### Running Tests

```bash
uv run pytest                              # Run all tests
uv run pytest tests/test_basic_decompose.py  # Run specific file
uv run pytest -v                           # Verbose output
```

For advanced testing options, see [docs/development.md#testing](docs/development.md#testing).

## Pull Request Process

### Before Submitting

Ensure your changes are ready:

1. ✅ All tests pass (`uv run pytest`)
2. ✅ Code is formatted (`uv run ruff format src/ tests/ vendor/`)
3. ✅ No linting errors (`uv run ruff check src/ tests/ vendor/`)
4. ✅ Type checking passes (see [docs/development.md#type-checking](docs/development.md#type-checking))
5. ✅ Documentation updated if needed
6. ✅ Commits follow conventional format

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
