"""Basic integration tests for LayerD decomposition."""

from pathlib import Path

import pytest
from PIL import Image

from layerd import LayerD

# Test with sample image
TEST_IMAGE_PATH = Path(__file__).parent.parent / "data" / "test_image_2.png"


@pytest.mark.parametrize(
    "fg_refine,bg_refine",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_decompose(fg_refine: bool, bg_refine: bool, matting_process_size: tuple, save_images: bool) -> None:
    """Test LayerD decompose with various refine options."""
    # Load test image
    image = Image.open(TEST_IMAGE_PATH)

    # Initialize LayerD model
    layerd = LayerD(
        matting_hf_card="cyberagent/layerd-birefnet",
        matting_process_size=matting_process_size,  # Use configurable size from fixture
        use_unblend=True,  # Fixed to True as requested
        fg_refine=fg_refine,
        bg_refine=bg_refine,
    ).to("cpu")

    # Decompose the image
    layers = layerd.decompose(image)

    # Basic assertions
    assert isinstance(layers, list), (
        f"decompose() should return a list for fg_refine={fg_refine}, bg_refine={bg_refine}"
    )
    assert len(layers) >= 1, f"Should return at least one layer for fg_refine={fg_refine}, bg_refine={bg_refine}"

    # Check each layer
    for i, layer in enumerate(layers):
        assert isinstance(layer, Image.Image), f"Layer {i} should be a PIL Image"
        assert layer.mode == "RGBA", f"Layer {i} should be in RGBA mode, got {layer.mode}"
        assert layer.size == image.size, f"Layer {i} size should match input image"

    # Save images if requested (following the same pattern as tools/infer.py)
    if save_images:
        # Create output directory with test configuration in the name
        output_dir = Path(__file__).parent / "output" / f"fg_{fg_refine}_bg_{bg_refine}_size_{matting_process_size[0]}x{matting_process_size[1]}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each layer with zero-padded index like infer.py
        for i, layer in enumerate(layers):
            layer.save(output_dir / f"{i:04d}.png")

        print(f"✓ Images saved to: {output_dir}")

    print(f"✓ fg_refine={fg_refine}, bg_refine={bg_refine}: returned {len(layers)} layers")
