"""Tests for types and utility functions (boxes, masks)."""

import numpy as np
import pytest
from PIL import Image

from layerd.types import BoundingBox, Element
from layerd.utils.boxes import apply_mask, assign_masks_to_groups, crop_image_with_bbox, mask_to_bbox
from layerd.utils.masks import compute_iou, compute_overlap_ratio, vertices_to_mask


# ===========================
# Type Tests
# ===========================


def test_bounding_box_type():
    """Test BoundingBox TypedDict creation and access."""
    bbox: BoundingBox = {
        "x_min": 10,
        "y_min": 20,
        "x_max": 100,
        "y_max": 80,
    }

    assert bbox["x_min"] == 10
    assert bbox["y_min"] == 20
    assert bbox["x_max"] == 100
    assert bbox["y_max"] == 80

    # Test width/height calculations
    width = bbox["x_max"] - bbox["x_min"]
    height = bbox["y_max"] - bbox["y_min"]
    assert width == 90
    assert height == 60


def test_element_type():
    """Test Element TypedDict creation with all fields."""
    bbox: BoundingBox = {
        "x_min": 0,
        "y_min": 0,
        "x_max": 50,
        "y_max": 50,
    }

    img = Image.new("RGBA", (50, 50), (255, 0, 0, 255))

    element: Element = {
        "id": 1,
        "type": "text",
        "image": img,
        "box": bbox,
    }

    assert element["id"] == 1
    assert element["type"] == "text"
    assert isinstance(element["image"], Image.Image)
    assert element["box"] == bbox


def test_element_types():
    """Test Element type field accepts all valid types."""
    for element_type in ["text", "vector", "image"]:
        element: Element = {
            "id": 1,
            "type": element_type,
            "image": Image.new("RGBA", (10, 10)),
            "box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
        }
        assert element["type"] == element_type


# ===========================
# Mask Utility Tests
# ===========================


def test_vertices_to_mask():
    """Test polygon vertices to mask conversion."""
    vertices = [
        {"x": "10", "y": "10"},
        {"x": "50", "y": "10"},
        {"x": "50", "y": "50"},
        {"x": "10", "y": "50"},
    ]
    mask = vertices_to_mask(vertices, canvas_size=(100, 100))

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8

    # Inside polygon should be 255
    assert mask[30, 30] == 255

    # Outside polygon should be 0
    assert mask[5, 5] == 0
    assert mask[80, 80] == 0


def test_vertices_to_mask_triangle():
    """Test triangle polygon conversion."""
    vertices = [
        {"x": "50", "y": "10"},
        {"x": "90", "y": "90"},
        {"x": "10", "y": "90"},
    ]
    mask = vertices_to_mask(vertices, canvas_size=(100, 100))

    # Center of triangle should be filled
    assert mask[60, 50] == 255

    # Top corners outside triangle
    assert mask[5, 5] == 0
    assert mask[5, 95] == 0


def test_compute_iou():
    """Test IoU calculation between two masks."""
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[20:80, 30:70] = True  # 60 * 40 = 2400 pixels

    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[40:90, 30:70] = True  # 50 * 40 = 2000 pixels

    # Intersection: [40:80, 30:70] = 40 * 40 = 1600 pixels
    # Union: 2400 + 2000 - 1600 = 2800 pixels
    # IoU: 1600 / 2800 ≈ 0.571
    iou = compute_iou(mask1, mask2)
    expected_iou = 1600 / 2800
    assert abs(iou - expected_iou) < 0.001


def test_compute_iou_edge_cases():
    """Test IoU edge cases."""
    # Empty masks
    empty1 = np.zeros((100, 100), dtype=bool)
    empty2 = np.zeros((100, 100), dtype=bool)
    assert compute_iou(empty1, empty2) == 0.0

    # No overlap
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[0:50, 0:50] = True

    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[50:100, 50:100] = True

    assert compute_iou(mask1, mask2) == 0.0

    # Perfect overlap
    mask3 = np.zeros((100, 100), dtype=bool)
    mask3[20:80, 30:70] = True

    mask4 = mask3.copy()

    assert compute_iou(mask3, mask4) == 1.0


def test_compute_iou_uint8_dtype():
    """Test IoU with uint8 masks."""
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:80, 30:70] = 255

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[40:90, 30:70] = 200  # Non-255 value should still work

    iou = compute_iou(mask1, mask2)
    assert iou > 0.5  # Should compute correctly


def test_compute_overlap_ratio():
    """Test overlap ratio calculation."""
    # mask1: small centered square
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[40:60, 40:60] = True  # 20 * 20 = 400 pixels

    # mask2: large square covering entire image
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[20:80, 20:80] = True  # 60 * 60 = 3600 pixels

    # mask1 fully inside mask2, so overlap = 1.0
    overlap = compute_overlap_ratio(mask1, mask2)
    assert abs(overlap - 1.0) < 0.001

    # Reverse: mask2 inside mask1 (partially)
    # Intersection: 400 pixels
    # Area(mask2): 3600 pixels
    # Overlap: 400 / 3600 ≈ 0.111
    overlap_reverse = compute_overlap_ratio(mask2, mask1)
    expected = 400 / 3600
    assert abs(overlap_reverse - expected) < 0.001


def test_compute_overlap_ratio_edge_cases():
    """Test overlap ratio edge cases."""
    # Empty mask1
    empty = np.zeros((100, 100), dtype=bool)
    mask = np.ones((100, 100), dtype=bool)
    assert compute_overlap_ratio(empty, mask) == 0.0

    # No overlap
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[0:50, 0:50] = True

    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[50:100, 50:100] = True

    assert compute_overlap_ratio(mask1, mask2) == 0.0


# ===========================
# Box Utility Tests
# ===========================


def test_mask_to_bbox():
    """Test bbox extraction from mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True

    bbox = mask_to_bbox(mask)

    assert bbox["x_min"] == 30
    assert bbox["y_min"] == 20
    assert bbox["x_max"] == 70  # Exclusive
    assert bbox["y_max"] == 80  # Exclusive

    # Verify dimensions
    width = bbox["x_max"] - bbox["x_min"]
    height = bbox["y_max"] - bbox["y_min"]
    assert width == 40
    assert height == 60


def test_mask_to_bbox_uint8():
    """Test bbox extraction from uint8 mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 20:50] = 255

    bbox = mask_to_bbox(mask)

    assert bbox["x_min"] == 20
    assert bbox["y_min"] == 10
    assert bbox["x_max"] == 50
    assert bbox["y_max"] == 30


def test_mask_to_bbox_single_pixel():
    """Test bbox extraction from single pixel mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[50, 60] = True

    bbox = mask_to_bbox(mask)

    assert bbox["x_min"] == 60
    assert bbox["y_min"] == 50
    assert bbox["x_max"] == 61
    assert bbox["y_max"] == 51


def test_mask_to_bbox_empty_mask():
    """Test that empty mask raises assertion."""
    mask = np.zeros((100, 100), dtype=bool)

    with pytest.raises(AssertionError, match="mask must not be empty"):
        mask_to_bbox(mask)


def test_apply_mask_rgba():
    """Test mask application to RGBA image."""
    # Create red RGBA image
    image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))

    # Create circular-ish mask (center square)
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True

    masked_img = apply_mask(image, mask)

    assert masked_img.mode == "RGBA"
    assert masked_img.size == (100, 100)

    # Check that masked region is visible
    img_array = np.array(masked_img)
    assert img_array[50, 50, 3] == 255  # Center alpha should be 255

    # Check that unmasked region is transparent
    assert img_array[10, 10, 3] == 0  # Corner alpha should be 0


def test_apply_mask_rgb_conversion():
    """Test that RGB image is converted to RGBA."""
    # Create red RGB image
    image = Image.new("RGB", (100, 100), (255, 0, 0))

    mask = np.ones((100, 100), dtype=bool)

    masked_img = apply_mask(image, mask)

    assert masked_img.mode == "RGBA"
    assert masked_img.size == (100, 100)

    # Check that alpha channel was added
    img_array = np.array(masked_img)
    assert img_array.shape[2] == 4
    assert img_array[50, 50, 3] == 255  # Alpha should be 255 where mask is True


def test_apply_mask_uint8():
    """Test mask application with uint8 mask."""
    image = Image.new("RGBA", (50, 50), (0, 255, 0, 255))

    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 200  # Non-255 value should still work

    masked_img = apply_mask(image, mask)

    img_array = np.array(masked_img)
    assert img_array[25, 25, 3] > 0  # Inside mask
    assert img_array[5, 5, 3] == 0  # Outside mask


def test_crop_image_with_bbox():
    """Test image cropping with bbox."""
    # Create 100x100 image with red color
    image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))

    bbox: BoundingBox = {
        "x_min": 20,
        "y_min": 30,
        "x_max": 80,
        "y_max": 70,
    }

    cropped = crop_image_with_bbox(image, bbox)

    # Check dimensions
    assert cropped.size == (60, 40)  # width=80-20, height=70-30

    # Check color is preserved
    cropped_array = np.array(cropped)
    assert np.all(cropped_array[20, 30, :3] == [255, 0, 0])


def test_crop_image_with_bbox_full_image():
    """Test cropping entire image."""
    image = Image.new("RGBA", (50, 50), (0, 0, 255, 255))

    bbox: BoundingBox = {
        "x_min": 0,
        "y_min": 0,
        "x_max": 50,
        "y_max": 50,
    }

    cropped = crop_image_with_bbox(image, bbox)

    assert cropped.size == image.size


def test_assign_masks_to_groups():
    """Test mask assignment to groups."""
    # Create masks to assign
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[10:30, 10:30] = 255  # Top-left

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[50:70, 50:70] = 255  # Bottom-right

    mask3 = np.zeros((100, 100), dtype=np.uint8)
    mask3[15:25, 15:25] = 255  # Overlaps with mask1

    # Create group masks
    group1 = np.zeros((100, 100), dtype=np.uint8)
    group1[0:50, 0:50] = 255  # Top-left quadrant

    group2 = np.zeros((100, 100), dtype=np.uint8)
    group2[50:100, 50:100] = 255  # Bottom-right quadrant

    # Assign with 50% overlap threshold
    combined, unassigned = assign_masks_to_groups(
        [mask1, mask2, mask3],
        [group1, group2],
        overlap_threshold=0.5,
    )

    # Check combined masks
    assert len(combined) == 2
    assert len(unassigned) == 0

    # mask1 and mask3 should be assigned to group1
    assert np.any(combined[0][10:30, 10:30] > 0)  # mask1
    assert np.any(combined[0][15:25, 15:25] > 0)  # mask3

    # mask2 should be assigned to group2
    assert np.any(combined[1][50:70, 50:70] > 0)


def test_assign_masks_to_groups_unassigned():
    """Test that masks below threshold are unassigned."""
    # Create mask with low overlap
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[45:55, 45:55] = 255  # Center, low overlap with groups

    # Create non-overlapping group masks
    group1 = np.zeros((100, 100), dtype=np.uint8)
    group1[0:40, 0:40] = 255  # Top-left

    group2 = np.zeros((100, 100), dtype=np.uint8)
    group2[60:100, 60:100] = 255  # Bottom-right

    # Assign with high threshold (80%)
    combined, unassigned = assign_masks_to_groups(
        [mask1],
        [group1, group2],
        overlap_threshold=0.8,
    )

    # mask1 should be unassigned (overlap < 80% with any group)
    assert len(unassigned) == 1
    assert np.array_equal(unassigned[0], mask1)


def test_assign_masks_to_groups_best_match():
    """Test that masks are assigned to best matching group."""
    # Create mask that overlaps with both groups
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[40:60, 40:60] = 255  # Center

    # Group1: larger overlap
    group1 = np.zeros((100, 100), dtype=np.uint8)
    group1[30:70, 30:70] = 255  # Larger overlap with mask1

    # Group2: smaller overlap
    group2 = np.zeros((100, 100), dtype=np.uint8)
    group2[50:80, 50:80] = 255  # Smaller overlap

    combined, unassigned = assign_masks_to_groups(
        [mask1],
        [group1, group2],
        overlap_threshold=0.3,
    )

    # mask1 should be assigned to group1 (better match)
    assert np.any(combined[0][40:60, 40:60] > 0)
    assert not np.any(combined[1] > 0)


def test_assign_masks_to_groups_empty_masks_list():
    """Test that empty masks list raises assertion."""
    group1 = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(AssertionError, match="masks list must not be empty"):
        assign_masks_to_groups([], [group1], overlap_threshold=0.5)


# ===========================
# Integration Tests
# ===========================


def test_mask_to_bbox_and_crop():
    """Test integration of mask_to_bbox and crop_image_with_bbox."""
    # Create mask
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True

    # Extract bbox
    bbox = mask_to_bbox(mask)

    # Create image and crop
    image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    cropped = crop_image_with_bbox(image, bbox)

    # Verify cropped dimensions match bbox
    expected_width = bbox["x_max"] - bbox["x_min"]
    expected_height = bbox["y_max"] - bbox["y_min"]

    assert cropped.size == (expected_width, expected_height)


def test_apply_mask_and_crop():
    """Test integration of apply_mask and crop_image_with_bbox."""
    # Create image
    image = Image.new("RGBA", (100, 100), (0, 255, 0, 255))

    # Create mask
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True

    # Apply mask
    masked_img = apply_mask(image, mask)

    # Extract bbox and crop
    bbox = mask_to_bbox(mask)
    cropped = crop_image_with_bbox(masked_img, bbox)

    # Verify cropped image is 50x50
    assert cropped.size == (50, 50)

    # Verify alpha channel is correct
    cropped_array = np.array(cropped)
    assert np.all(cropped_array[:, :, 3] == 255)  # All visible in cropped region
