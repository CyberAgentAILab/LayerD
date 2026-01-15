"""Tests for element classification."""

import numpy as np
from PIL import Image, ImageDraw

from layerd.classification import EntropyLabeler, GradientAwareLabeler, compute_entropy
from layerd.types import BoundingBox, Element


def create_solid_color_element(elem_id: int, size: tuple[int, int], color: tuple[int, int, int, int]) -> Element:
    """Create element with solid color (low entropy)."""
    img = Image.new("RGBA", size, color)
    return Element(
        id=elem_id, type="image", image=img, box=BoundingBox(x_min=0, y_min=0, x_max=size[0], y_max=size[1])
    )


def create_noise_element(elem_id: int, size: tuple[int, int]) -> Element:
    """Create element with random noise (high entropy)."""
    arr = np.random.randint(0, 256, (size[1], size[0], 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # Full alpha
    img = Image.fromarray(arr, "RGBA")
    return Element(
        id=elem_id, type="image", image=img, box=BoundingBox(x_min=0, y_min=0, x_max=size[0], y_max=size[1])
    )


def create_gradient_element(elem_id: int, size: tuple[int, int]) -> Element:
    """Create element with gradient (low entropy but should stay as image)."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Create horizontal gradient
    for x in range(size[0]):
        color_val = int(255 * x / size[0])
        draw.line([(x, 0), (x, size[1])], fill=(color_val, color_val, color_val, 255))

    return Element(
        id=elem_id, type="image", image=img, box=BoundingBox(x_min=0, y_min=0, x_max=size[0], y_max=size[1])
    )


def test_compute_entropy_solid_color() -> None:
    """Test entropy computation for solid color."""
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    entropy = compute_entropy(img)
    assert entropy < 1.0, "Solid color should have very low entropy"


def test_compute_entropy_noise() -> None:
    """Test entropy computation for noise."""
    arr = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    img = Image.fromarray(arr, "RGBA")
    entropy = compute_entropy(img)
    assert entropy > 5.0, "Random noise should have high entropy"


def test_entropy_labeler_vector_classification() -> None:
    """Test that solid colors are classified as vectors."""
    elements = [
        create_solid_color_element(1, (50, 50), (255, 0, 0, 255)),
        create_solid_color_element(2, (50, 50), (0, 255, 0, 255)),
    ]

    labeler = EntropyLabeler(threshold=5.0)
    result = labeler.classify(elements)

    assert len(result) == 2
    assert all(elem["type"] == "vector" for elem in result), "Solid colors should be classified as vector"


def test_entropy_labeler_image_classification() -> None:
    """Test that high-entropy content is classified as images."""
    elements = [
        create_noise_element(1, (50, 50)),
        create_noise_element(2, (50, 50)),
    ]

    labeler = EntropyLabeler(threshold=5.0)
    result = labeler.classify(elements)

    assert len(result) == 2
    assert all(elem["type"] == "image" for elem in result), "Noise should be classified as image"


def test_entropy_labeler_preserves_text() -> None:
    """Test that text elements are not modified."""
    img = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
    elements = [Element(id=1, type="text", image=img, box=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50))]

    labeler = EntropyLabeler(threshold=5.0)
    result = labeler.classify(elements)

    assert result[0]["type"] == "text", "Text elements should not be modified"


def test_gradient_aware_labeler_gradient_detection() -> None:
    """Test that gradients are detected and kept as images."""
    elements = [create_gradient_element(1, (100, 100))]

    labeler = GradientAwareLabeler(entropy_threshold=5.0, gradient_threshold=0.3)
    result = labeler.classify(elements)

    assert result[0]["type"] == "image", "Gradients should be kept as raster images"


def test_gradient_aware_labeler_solid_vector() -> None:
    """Test that solid colors are classified as vectors."""
    elements = [create_solid_color_element(1, (50, 50), (255, 0, 0, 255))]

    labeler = GradientAwareLabeler(entropy_threshold=5.0, gradient_threshold=0.3)
    result = labeler.classify(elements)

    assert result[0]["type"] == "vector", "Solid colors without gradients should be vectors"


def test_gradient_aware_labeler_noise_image() -> None:
    """Test that noisy images remain as images."""
    elements = [create_noise_element(1, (50, 50))]

    labeler = GradientAwareLabeler(entropy_threshold=5.0, gradient_threshold=0.3)
    result = labeler.classify(elements)

    assert result[0]["type"] == "image", "Noisy content should remain as image"


def test_gradient_aware_labeler_custom_thresholds() -> None:
    """Test labeler with custom thresholds."""
    elements = [
        create_solid_color_element(1, (50, 50), (255, 0, 0, 255)),
        create_gradient_element(2, (100, 100)),
    ]

    # Very strict gradient threshold
    labeler = GradientAwareLabeler(entropy_threshold=5.0, gradient_threshold=1.0)
    result = labeler.classify(elements)

    assert result[0]["type"] == "vector", "Solid color should be vector"
    # Gradient might not be detected with high threshold, but should still work


def test_classification_preserves_element_structure() -> None:
    """Test that classification preserves all element fields."""
    img = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
    original = Element(id=42, type="image", image=img, box=BoundingBox(x_min=10, y_min=20, x_max=60, y_max=70))

    labeler = EntropyLabeler()
    result = labeler.classify([original])

    assert result[0]["id"] == 42
    assert result[0]["image"].size == (50, 50)
    assert result[0]["box"]["x_min"] == 10
    assert result[0]["box"]["y_min"] == 20
    assert result[0]["box"]["x_max"] == 60
    assert result[0]["box"]["y_max"] == 70
