"""
Pytest configuration for LayerD tests.

This file provides command-line options and fixtures for tests.
"""
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--save-images",
        action="store_true",
        default=False,
        help="Save output images during testing (default: False)"
    )
    parser.addoption(
        "--matting-process-size",
        nargs=2,
        type=int,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "Matting process size in pixels as WIDTH HEIGHT (default: 256 256). "
            "Smaller values can speed up CPU tests at the cost of output quality; "
            "for full-resolution testing, use the dimensions of your input images "
            "(for example, 3840 2160 for 4K)."
        )
    )


@pytest.fixture
def save_images(request: pytest.FixtureRequest) -> bool:
    """Fixture to access the save-images flag."""
    return request.config.getoption("--save-images")


@pytest.fixture
def matting_process_size(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Fixture to access the matting-process-size option as a tuple."""
    size = request.config.getoption("--matting-process-size")
    return tuple(size)


@pytest.fixture(scope="module")
def layerd_model(matting_process_size: tuple[int, int]) -> "LayerD":  # noqa: F821
    """Create LayerD model once per module with smaller size for faster CPU tests.

    This fixture creates a LayerD model with optimized test parameters once
    per module. The model loads ~1GB BiRefNet weights, so reusing the same
    instance significantly speeds up test runs.

    The default matting_process_size is 256x256 instead of the default 1024x1024,
    which provides 4x speedup on CPU while maintaining test coverage.
    For full-resolution testing, use: pytest --matting-process-size 1024 1024

    Returns:
        LayerD: A LayerD model configured for testing on CPU.
    """
    from layerd import LayerD

    return LayerD(
        matting_hf_card="cyberagent/layerd-birefnet",
        matting_process_size=matting_process_size,
    ).to("cpu")