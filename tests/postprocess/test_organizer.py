"""Tests for LayerOrganizer."""

from PIL import Image, ImageDraw

from layerd.postprocess import LayerOrganizer
from layerd.types import BoundingBox


def create_dummy_layer(width: int, height: int, shapes: list[tuple[str, tuple[int, int, int, int]]]) -> Image.Image:
    """Create dummy RGBA layer with shapes.

    Args:
        width: Image width
        height: Image height
        shapes: List of (shape_type, bbox) tuples
                shape_type: "rect" or "circle"
                bbox: (x_min, y_min, x_max, y_max)

    Returns:
        RGBA PIL Image
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for shape_type, bbox in shapes:
        if shape_type == "rect":
            draw.rectangle(bbox, fill=(255, 100, 100, 255))
        elif shape_type == "circle":
            draw.ellipse(bbox, fill=(100, 100, 255, 255))

    return img


def test_organizer_basic_no_ocr() -> None:
    """Test basic LayerOrganizer functionality without OCR."""
    # Create dummy layers
    layer1 = create_dummy_layer(
        512,
        512,
        [
            ("rect", (50, 50, 150, 100)),  # Shape 1
            ("circle", (300, 300, 400, 400)),  # Shape 2
        ],
    )
    layer2 = create_dummy_layer(512, 512, [("rect", (200, 200, 300, 250))])  # Shape 3

    layers = [layer1, layer2]

    # Run organizer without OCR
    organizer = LayerOrganizer()
    elements = organizer.organize(layers)

    # Basic assertions
    assert len(elements) > 0, "Should return at least one element"
    assert all("id" in elem for elem in elements), "All elements should have 'id'"
    assert all("type" in elem for elem in elements), "All elements should have 'type'"
    assert all("image" in elem for elem in elements), "All elements should have 'image'"
    assert all("box" in elem for elem in elements), "All elements should have 'box'"

    # Without OCR, all elements should be "image" type
    types = [elem["type"] for elem in elements]
    assert all(t == "image" for t in types), "All elements should be 'image' type without OCR"


def test_organizer_with_ocr() -> None:
    """Test LayerOrganizer with OCR result."""
    # Create layer with text region
    layer = create_dummy_layer(256, 256, [("rect", (50, 50, 150, 100))])

    # Create OCR result that matches the rect
    ocr_result = {
        "image_size": (256, 256),
        "blocks": [
            {
                "text": "Sample text",
                "bbox": BoundingBox(x_min=50, y_min=50, x_max=150, y_max=100),
            }
        ],
    }

    organizer = LayerOrganizer(overlap_threshold=0.9)
    elements = organizer.organize([layer], ocr_result)

    # Should have exactly one text element
    text_elements = [e for e in elements if e["type"] == "text"]
    assert len(text_elements) == 1, f"Should have exactly one text element, got {len(text_elements)}"

    # Verify the cropped image is RGBA
    assert text_elements[0]["image"].mode == "RGBA", "Text element image should be RGBA"


def test_organizer_no_ocr_match() -> None:
    """Test when OCR doesn't match any layer regions."""
    # Create layer with shape
    layer = create_dummy_layer(256, 256, [("circle", (50, 50, 150, 150))])

    # Create OCR result far from the shape
    ocr_result = {
        "image_size": (256, 256),
        "blocks": [
            {
                "text": "Sample text",
                "bbox": BoundingBox(x_min=200, y_min=200, x_max=250, y_max=250),
            }
        ],
    }

    organizer = LayerOrganizer(overlap_threshold=0.9)
    elements = organizer.organize([layer], ocr_result)

    # All elements should be non-text type (image)
    assert all(e["type"] == "image" for e in elements), "All elements should be 'image' type"


def test_organizer_output_format() -> None:
    """Test output format consistency."""
    layer = create_dummy_layer(128, 128, [("rect", (20, 20, 60, 60))])

    organizer = LayerOrganizer()
    elements = organizer.organize([layer])

    for elem in elements:
        # Check Element structure
        assert isinstance(elem["id"], int), f"Element id should be int, got {type(elem['id'])}"
        assert elem["type"] in ["text", "image", "vector"], f"Invalid element type: {elem['type']}"
        assert isinstance(elem["image"], Image.Image), "Element image should be PIL Image"
        assert "x_min" in elem["box"], "Element box should have 'x_min'"
        assert "y_min" in elem["box"], "Element box should have 'y_min'"
        assert "x_max" in elem["box"], "Element box should have 'x_max'"
        assert "y_max" in elem["box"], "Element box should have 'y_max'"

        # Check bounding box validity
        box = elem["box"]
        assert box["x_min"] < box["x_max"], "x_min should be less than x_max"
        assert box["y_min"] < box["y_max"], "y_min should be less than y_max"


def test_organizer_callable() -> None:
    """Test that LayerOrganizer can be called as a function."""
    layer = create_dummy_layer(128, 128, [("rect", (20, 20, 60, 60))])

    organizer = LayerOrganizer()

    # Test both calling methods
    elements1 = organizer.organize([layer])
    elements2 = organizer([layer])

    assert len(elements1) == len(elements2)


def test_organizer_empty_layers() -> None:
    """Test organizer with fully transparent layers."""
    # Create completely transparent layer
    layer = Image.new("RGBA", (100, 100), (0, 0, 0, 0))

    organizer = LayerOrganizer()
    elements = organizer.organize([layer])

    # Should return empty list or handle gracefully
    assert isinstance(elements, list)


def test_organizer_with_polygon_ocr() -> None:
    """Test LayerOrganizer with polygon-based OCR result."""
    # Create layer with shape
    layer = create_dummy_layer(256, 256, [("rect", (50, 50, 150, 100))])

    # Create OCR result with polygon
    ocr_result = {
        "image_size": (256, 256),
        "blocks": [
            {
                "text": "Sample text",
                "bbox": BoundingBox(x_min=50, y_min=50, x_max=150, y_max=100),
                "polygon": [
                    {"x": 50, "y": 50},
                    {"x": 150, "y": 50},
                    {"x": 150, "y": 100},
                    {"x": 50, "y": 100},
                ],
            }
        ],
    }

    organizer = LayerOrganizer(overlap_threshold=0.9)
    elements = organizer.organize([layer], ocr_result)

    # Should have at least one element
    assert len(elements) > 0
    # Should have text element due to polygon match
    text_elements = [e for e in elements if e["type"] == "text"]
    assert len(text_elements) >= 1
