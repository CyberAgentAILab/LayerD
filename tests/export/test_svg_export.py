"""Tests for SVG export functionality."""

from pathlib import Path

from PIL import Image, ImageDraw

from layerd.export import SVGBuilder, SVGParser
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
    img = Image.new("RGBA", (width, height), (255, 100, 100, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(0, 0, 0, 255), width=2)

    return Element(
        id=elem_id,
        type=elem_type,
        image=img,
        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
    )


def test_svg_builder_base64() -> None:
    """Test SVG export with base64 image embedding."""
    elements = [
        create_test_element(1, "text", (10, 10, 110, 60)),
        create_test_element(2, "vector", (150, 50, 250, 150)),
    ]

    builder = SVGBuilder(image_mode="base64")
    svg_content = builder(elements, canvas_size=(300, 200))

    # Verify SVG structure
    assert "<svg" in svg_content
    assert 'width="300"' in svg_content
    assert 'height="200"' in svg_content
    assert "</svg>" in svg_content

    # Verify base64 embedding
    assert "data:image/png;base64," in svg_content

    # Verify metadata
    assert 'data-type="text"' in svg_content
    assert 'data-type="vector"' in svg_content
    assert 'data-id="1"' in svg_content
    assert 'data-id="2"' in svg_content


def test_svg_builder_external_images(tmp_path: Path) -> None:
    """Test SVG export with external image files."""
    elements = [
        create_test_element(1, "text", (20, 20, 120, 70)),
        create_test_element(2, "image", (150, 50, 250, 150)),
    ]

    image_dir = tmp_path / "images"
    builder = SVGBuilder(image_mode="external", image_dir=str(image_dir))
    svg_content = builder(elements, canvas_size=(300, 200))

    # Verify SVG structure
    assert "<svg" in svg_content
    assert "</svg>" in svg_content

    # Should NOT have base64 data
    assert "data:image/png;base64," not in svg_content

    # Should have external references
    assert "./images/" in svg_content

    # Check that image files were created
    assert (image_dir / "1_text.png").exists()
    assert (image_dir / "2_image.png").exists()


def test_svg_parser_base64() -> None:
    """Test SVG parsing with base64 images."""
    # Create elements
    original_elements = [
        create_test_element(1, "text", (10, 10, 110, 60)),
        create_test_element(2, "vector", (150, 50, 250, 150)),
    ]

    # Export to SVG
    builder = SVGBuilder(image_mode="base64")
    svg_content = builder(original_elements, canvas_size=(300, 200))

    # Parse back
    parser = SVGParser()
    parsed_elements = parser(svg_content)

    # Verify round-trip
    assert len(parsed_elements) == len(original_elements)

    for orig, parsed in zip(original_elements, parsed_elements):
        assert orig["id"] == parsed["id"]
        assert orig["type"] == parsed["type"]
        assert orig["box"] == parsed["box"]
        assert orig["image"].size == parsed["image"].size


def test_svg_parser_external_images(tmp_path: Path) -> None:
    """Test SVG parsing with external image files."""
    # Create elements
    original_elements = [
        create_test_element(1, "text", (20, 20, 120, 70)),
        create_test_element(2, "image", (150, 50, 250, 150)),
    ]

    # Export to SVG with external images
    image_dir = tmp_path / "images"
    builder = SVGBuilder(image_mode="external", image_dir=str(image_dir))
    svg_content = builder(original_elements, canvas_size=(300, 200))

    # Save SVG file
    svg_path = tmp_path / "test.svg"
    with open(svg_path, "w") as f:
        f.write(svg_content)

    # Parse back
    parser = SVGParser()
    parsed_elements = parser(svg_content, svg_path=str(svg_path))

    # Verify round-trip
    assert len(parsed_elements) == len(original_elements)

    for orig, parsed in zip(original_elements, parsed_elements):
        assert orig["id"] == parsed["id"]
        assert orig["type"] == parsed["type"]
        assert orig["box"] == parsed["box"]
        assert orig["image"].size == parsed["image"].size


def test_svg_builder_callable() -> None:
    """Test that SVGBuilder can be called as a function."""
    elements = [create_test_element(1, "text", (10, 10, 110, 60))]

    builder = SVGBuilder()

    # Test both calling methods
    svg1 = builder.export(elements, canvas_size=(200, 100))
    svg2 = builder(elements, canvas_size=(200, 100))

    assert svg1 == svg2


def test_svg_builder_empty_elements() -> None:
    """Test SVG export with no elements."""
    builder = SVGBuilder()
    svg_content = builder([], canvas_size=(300, 200))

    assert "<svg" in svg_content
    assert 'width="300"' in svg_content
    assert 'height="200"' in svg_content
    assert "</svg>" in svg_content
    # Should have no image elements
    assert "<image" not in svg_content
