"""Bounding box and image operations for postprocessing.

This module provides utilities for working with bounding boxes, including
mask-to-bbox conversion, image cropping, and mask-to-group assignment.

Example:
    >>> import numpy as np
    >>> from PIL import Image
    >>> from layerd.utils.boxes import mask_to_bbox, crop_image_with_bbox
    >>>
    >>> # Create a mask and extract bounding box
    >>> mask = np.zeros((100, 100), dtype=bool)
    >>> mask[20:80, 30:70] = True
    >>> bbox = mask_to_bbox(mask)
    >>> # {'x_min': 30, 'y_min': 20, 'x_max': 70, 'y_max': 80}
    >>>
    >>> # Crop image using bbox
    >>> image = Image.new("RGBA", (100, 100))
    >>> cropped = crop_image_with_bbox(image, bbox)
"""

import numpy as np
from PIL import Image

from layerd.types import BoundingBox
from layerd.utils.masks import compute_overlap_ratio


def mask_to_bbox(mask: np.ndarray) -> BoundingBox:
    """Extract tight bounding box from a binary mask.

    Args:
        mask: Binary mask as numpy array with shape (H, W).
            Can be bool or uint8 dtype.

    Returns:
        BoundingBox dictionary with {x_min, y_min, x_max, y_max}.
        Coordinates use exclusive upper bounds (x_max, y_max).

    Raises:
        AssertionError: If mask is empty (all zeros).

    Example:
        >>> mask = np.zeros((100, 100), dtype=bool)
        >>> mask[20:80, 30:70] = True
        >>> bbox = mask_to_bbox(mask)
        >>> bbox
        {'x_min': 30, 'y_min': 20, 'x_max': 70, 'y_max': 80}
        >>> width = bbox['x_max'] - bbox['x_min']  # 40
        >>> height = bbox['y_max'] - bbox['y_min']  # 60
    """
    # Convert to bool if needed
    if mask.dtype == np.uint8:
        mask_bool = mask > 0
    else:
        mask_bool = mask.astype(bool)

    # Find non-zero positions
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)

    assert rows.any() and cols.any(), "mask must not be empty"

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return BoundingBox(
        x_min=int(x_min),
        y_min=int(y_min),
        x_max=int(x_max) + 1,  # +1 for exclusive upper bound
        y_max=int(y_max) + 1,
    )


def apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply binary mask to image alpha channel.

    Multiplies the image's alpha channel by the mask, making masked-out
    regions transparent. If the image is RGB, converts to RGBA first.

    Args:
        image: PIL Image (RGB or RGBA).
        mask: Binary mask as numpy array with shape (H, W).
            Can be bool or uint8 dtype.

    Returns:
        PIL Image in RGBA mode with mask applied to alpha channel.

    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))  # Red
        >>> mask = np.zeros((100, 100), dtype=bool)
        >>> mask[25:75, 25:75] = True  # Center square
        >>> masked_img = apply_mask(image, mask)
        >>> # Only center square is visible, rest is transparent
    """
    img_array = np.array(image)
    mask_bool = mask > 0 if mask.dtype == np.uint8 else mask.astype(bool)

    # Apply mask to alpha channel
    if img_array.shape[2] == 4:
        img_array[:, :, 3] = img_array[:, :, 3] * mask_bool
    else:
        # Convert RGB to RGBA if needed
        alpha = np.ones(img_array.shape[:2], dtype=np.uint8) * 255
        alpha = alpha * mask_bool
        img_array = np.dstack([img_array, alpha])

    return Image.fromarray(img_array, mode="RGBA")


def crop_image_with_bbox(image: Image.Image, bbox: BoundingBox) -> Image.Image:
    """Crop PIL image to the specified bounding box.

    Args:
        image: PIL Image to crop.
        bbox: BoundingBox dictionary with {x_min, y_min, x_max, y_max}.

    Returns:
        Cropped PIL Image.

    Example:
        >>> from PIL import Image
        >>> image = Image.new("RGBA", (100, 100))
        >>> bbox = {'x_min': 20, 'y_min': 30, 'x_max': 80, 'y_max': 70}
        >>> cropped = crop_image_with_bbox(image, bbox)
        >>> cropped.size  # (60, 40)
    """
    return image.crop((bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]))


def assign_masks_to_groups(
    masks: list[np.ndarray],
    group_masks: list[np.ndarray],
    overlap_threshold: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Assign masks to groups based on overlap ratio.

    Each mask is assigned to the group with the highest overlap ratio,
    provided the overlap exceeds the threshold. Masks that don't meet
    the threshold for any group are returned as unassigned.

    Args:
        masks: List of masks to assign (e.g., connected components).
            All masks must have the same shape.
        group_masks: List of group masks (e.g., OCR text blocks).
            Must have the same shape as masks.
        overlap_threshold: Minimum overlap ratio in [0, 1] for assignment.

    Returns:
        Tuple of (combined_masks, unassigned_masks):
            - combined_masks: List of combined masks for each group (same length as group_masks)
            - unassigned_masks: List of masks that didn't meet threshold for any group

    Raises:
        AssertionError: If masks list is empty.

    Example:
        >>> import numpy as np
        >>> # Create masks to assign
        >>> mask1 = np.zeros((100, 100), dtype=np.uint8)
        >>> mask1[10:30, 10:30] = 255
        >>> mask2 = np.zeros((100, 100), dtype=np.uint8)
        >>> mask2[50:70, 50:70] = 255
        >>>
        >>> # Create group masks
        >>> group1 = np.zeros((100, 100), dtype=np.uint8)
        >>> group1[0:50, 0:50] = 255
        >>>
        >>> # Assign with 50% overlap threshold
        >>> combined, unassigned = assign_masks_to_groups([mask1, mask2], [group1], 0.5)
        >>> # mask1 assigned to group1 (high overlap)
        >>> # mask2 unassigned (low overlap with group1)
    """
    assert len(masks) > 0, "masks list must not be empty"

    # Initialize combined masks for each group
    group_combined_masks = [np.zeros_like(masks[0], dtype=np.uint8) for _ in group_masks]
    unassigned_masks: list[np.ndarray] = []

    # Assign each mask to the best matching group
    for mask in masks:
        best_group_idx = -1
        best_overlap = 0.0

        for group_idx, group_mask in enumerate(group_masks):
            overlap = compute_overlap_ratio(mask, group_mask)
            if overlap >= overlap_threshold and overlap > best_overlap:
                best_overlap = overlap
                best_group_idx = group_idx

        # Assign to best group or unassigned
        if best_group_idx >= 0:
            group_combined_masks[best_group_idx] = np.maximum(group_combined_masks[best_group_idx], mask)
        else:
            unassigned_masks.append(mask)

    return group_combined_masks, unassigned_masks


__all__ = ["mask_to_bbox", "apply_mask", "crop_image_with_bbox", "assign_masks_to_groups"]
