"""Tests for PSD export functionality."""

from PIL import Image, ImageDraw

from layerd.export import PSDBuilder
from layerd.types import BoundingBox, Element


def create_test_element(elem_id: int, elem_type: str, bbox: tuple[int, int, int, int]) -> Element:
    """Create a test element with a simple colored rectangle.

    Args:
        elem_id: Element ID
        elem_type: Element type ("text", "image", or "vector")
        bbox: Bounding box as (x_min, y_min, x_max, y_max)

    Returns:
        Test Element with RGBA image
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Create simple colored image
    colors = {"text": (255, 100, 100, 255), "image": (100, 255, 100, 255), "vector": (100, 100, 255, 255)}
    color = colors.get(elem_type, (128, 128, 128, 255))

    img = Image.new("RGBA", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(0, 0, 0, 255), width=2)

    return Element(
        id=elem_id,
        type=elem_type,
        image=img,
        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
    )


def test_psd_builder_basic() -> None:
    """Test basic PSD export."""
    elements = [
        create_test_element(1, "text", (10, 10, 110, 60)),
        create_test_element(2, "vector", (150, 50, 250, 150)),
    ]

    builder = PSDBuilder()
    psd_bytes = builder(elements, canvas_size=(300, 200))

    # Verify we got bytes
    assert isinstance(psd_bytes, bytes)
    assert len(psd_bytes) > 0

    # Verify PSD header signature (8BPS)
    assert psd_bytes[:4] == b"8BPS"


def test_psd_builder_compression_options() -> None:
    """Test PSD export with different compression options."""
    elements = [create_test_element(1, "text", (20, 20, 120, 70))]

    # Test different compression methods
    for compression in ["raw", "rle", "zip"]:
        builder = PSDBuilder(compression=compression)
        psd_bytes = builder(elements, canvas_size=(200, 100))

        assert isinstance(psd_bytes, bytes)
        assert len(psd_bytes) > 0
        assert psd_bytes[:4] == b"8BPS"


def test_psd_builder_color_depth() -> None:
    """Test PSD export with 8-bit color depth."""
    elements = [create_test_element(1, "image", (20, 20, 120, 70))]

    # Test 8-bit depth (default and most common)
    # Note: 16 and 32-bit depths have issues with psd-tools library
    builder = PSDBuilder(color_depth=8)
    psd_bytes = builder(elements, canvas_size=(200, 100))

    assert isinstance(psd_bytes, bytes)
    assert len(psd_bytes) > 0
    assert psd_bytes[:4] == b"8BPS"


def test_psd_builder_callable() -> None:
    """Test that PSDBuilder can be called as a function."""
    elements = [create_test_element(1, "text", (10, 10, 110, 60))]

    builder = PSDBuilder()

    # Test both calling methods
    psd1 = builder.export(elements, canvas_size=(200, 100))
    psd2 = builder(elements, canvas_size=(200, 100))

    assert psd1 == psd2


def test_psd_builder_multiple_layers() -> None:
    """Test PSD export with multiple layers."""
    elements = [
        create_test_element(1, "text", (10, 10, 100, 50)),
        create_test_element(2, "vector", (120, 20, 220, 120)),
        create_test_element(3, "image", (50, 150, 150, 250)),
    ]

    builder = PSDBuilder()
    psd_bytes = builder(elements, canvas_size=(300, 300))

    assert isinstance(psd_bytes, bytes)
    assert len(psd_bytes) > 0
    assert psd_bytes[:4] == b"8BPS"


def test_psd_builder_empty_elements() -> None:
    """Test PSD export with no elements."""
    builder = PSDBuilder()
    psd_bytes = builder([], canvas_size=(300, 200))

    assert isinstance(psd_bytes, bytes)
    assert len(psd_bytes) > 0
    assert psd_bytes[:4] == b"8BPS"
