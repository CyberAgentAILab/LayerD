"""Mask operations for postprocessing.

This module provides utilities for working with binary masks, including
polygon conversion, IoU computation, and overlap calculations.

Example:
    >>> import numpy as np
    >>> from layerd.utils.masks import compute_iou, compute_overlap_ratio
    >>>
    >>> # Create two overlapping masks
    >>> mask1 = np.zeros((100, 100), dtype=np.uint8)
    >>> mask1[20:80, 30:70] = 255
    >>> mask2 = np.zeros((100, 100), dtype=np.uint8)
    >>> mask2[40:90, 30:70] = 255
    >>>
    >>> # Compute IoU
    >>> iou = compute_iou(mask1, mask2)  # Intersection / Union
    >>> overlap = compute_overlap_ratio(mask1, mask2)  # Intersection / Area(mask1)
"""

import cv2
import numpy as np


def vertices_to_mask(vertices: list[dict[str, str]], canvas_size: tuple[int, int]) -> np.ndarray:
    """Convert polygon vertices to binary mask.

    Args:
        vertices: List of vertices with {"x": str, "y": str} format.
            Coordinates are expected as string representations of integers.
        canvas_size: Tuple of (width, height) of the canvas.

    Returns:
        Binary mask as uint8 numpy array with shape (H, W).
        Foreground pixels have value 255, background pixels have value 0.

    Example:
        >>> vertices = [
        ...     {"x": "10", "y": "10"},
        ...     {"x": "50", "y": "10"},
        ...     {"x": "50", "y": "50"},
        ...     {"x": "10", "y": "50"}
        ... ]
        >>> mask = vertices_to_mask(vertices, canvas_size=(100, 100))
        >>> mask.shape  # (100, 100)
        >>> mask[30, 30]  # 255 (inside polygon)
    """
    width, height = canvas_size
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert vertices to numpy array
    points = np.array([[int(v["x"]), int(v["y"])] for v in vertices], dtype=np.int32)

    # Fill polygon
    cv2.fillPoly(mask, [points], (255,))

    return mask


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two masks.

    IoU = Area(Intersection) / Area(Union)

    Args:
        mask1: Binary mask as numpy array with shape (H, W).
            Can be bool or uint8 dtype.
        mask2: Binary mask as numpy array with shape (H, W).
            Can be bool or uint8 dtype.

    Returns:
        IoU value in range [0, 1].
        Returns 0.0 if both masks are empty.

    Example:
        >>> mask1 = np.zeros((100, 100), dtype=bool)
        >>> mask1[20:80, 30:70] = True
        >>> mask2 = np.zeros((100, 100), dtype=bool)
        >>> mask2[40:90, 30:70] = True
        >>> iou = compute_iou(mask1, mask2)
        >>> print(f"IoU: {iou:.2f}")
    """
    # Convert to bool
    m1 = mask1 > 0
    m2 = mask2 > 0

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_overlap_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute overlap ratio: intersection / area(mask1).

    Measures what fraction of mask1 overlaps with mask2.
    Unlike IoU, this is asymmetric: overlap_ratio(A, B) != overlap_ratio(B, A).

    Args:
        mask1: Binary mask as numpy array with shape (H, W).
            Can be bool or uint8 dtype.
        mask2: Binary mask as numpy array with shape (H, W).
            Can be bool or uint8 dtype.

    Returns:
        Overlap ratio in range [0, 1].
        Returns 0.0 if mask1 is empty.

    Example:
        >>> mask1 = np.zeros((100, 100), dtype=bool)
        >>> mask1[20:80, 30:70] = True  # Small mask
        >>> mask2 = np.zeros((100, 100), dtype=bool)
        >>> mask2[0:100, 0:100] = True  # Large mask covering entire image
        >>> overlap = compute_overlap_ratio(mask1, mask2)  # 1.0 (mask1 fully inside mask2)
        >>> overlap_reverse = compute_overlap_ratio(mask2, mask1)  # ~0.24 (only 24% of mask2 overlaps)
    """
    m1 = mask1 > 0
    m2 = mask2 > 0

    area1 = m1.sum()
    if area1 == 0:
        return 0.0

    intersection = np.logical_and(m1, m2).sum()

    return float(intersection / area1)


__all__ = ["vertices_to_mask", "compute_iou", "compute_overlap_ratio"]
