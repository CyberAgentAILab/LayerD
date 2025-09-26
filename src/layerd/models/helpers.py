import logging
from collections import Counter
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def expand_mask(mask: np.ndarray, kernel_size: tuple[int, int] | int = (5, 5)) -> np.ndarray:
    """Expand mask using dilation.
    If mask is bool, return bool mask; if uint8, return uint8 mask.
    """
    is_bool = mask.dtype == bool
    mask = mask.astype(np.uint8)
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    mask = cv2.dilate(mask, np.ones(kernel_size, np.uint8))
    return mask > 0 if is_bool else mask


def shrink_mask(mask: np.ndarray, kernel_size: tuple[int, int] | int = (5, 5)) -> np.ndarray:
    """Shrink mask using erosion.
    If mask is bool, return bool mask; if uint8, return uint8 mask.
    """
    is_bool = mask.dtype == bool
    mask = mask.astype(np.uint8)
    kernel_tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

    # Add padding to avoid border effects
    pad_h, pad_w = kernel_tuple[0] // 2, kernel_tuple[1] // 2
    padded_mask = cv2.copyMakeBorder(mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,))
    eroded_mask = cv2.erode(padded_mask, np.ones(kernel_tuple, np.uint8))
    mask = eroded_mask
    if pad_h > 0:
        mask = mask[pad_h:-pad_h]
    if pad_w > 0:
        mask = mask[:, pad_w:-pad_w]

    return mask > 0 if is_bool else mask


def divide_mask_to_connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Divide binary mask into connected components."""
    assert mask.dtype == bool, f"Expected bool mask, got {mask.dtype}"
    num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
    ccs = [labels_im == label for label in range(1, num_labels)]
    return ccs


def calc_gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    """Calculate gradient magnitude of a color image using Sobel operator."""
    grad_x = cv2.Sobel(rgb, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(rgb, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2).sum(axis=-1)
    return grad_magnitude


def get_colors_until_percentile(color_counts: Counter, percentile: float) -> np.ndarray:
    """Calculate percentile from Counter object without pandas."""
    # Sort and calculate cumulative percentages
    sorted_items = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    colors = np.array([c for c, _ in sorted_items])
    counts = np.array([cnt for _, cnt in sorted_items])

    cumulative_ratios = np.cumsum(counts) / counts.sum()
    cutoff_idx = np.searchsorted(cumulative_ratios, percentile) + 1

    return colors[:cutoff_idx].astype(np.uint8)


def find_flat_color_region_ccs(
    rgb_np: np.ndarray,
    mask: np.ndarray,
    max_num_colors: int,
    percentile: float = 0.99,
    th_overlap_ratio: float = 0.5,
    th_flat_area_ratio: float = 0.2,
    th_color_match_ratio: float = 0.85,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Divide mask into connected components and find flat color regions in each component."""

    ccs = divide_mask_to_connected_components(mask)
    color_masks = []
    palette = []
    for cc in ccs:
        _color_masks, _palette = find_flat_color_region(
            rgb_np=rgb_np,
            mask=cc,
            max_num_colors=max_num_colors,
            percentile=percentile,
            th_overlap_ratio=th_overlap_ratio,
            th_flat_area_ratio=th_flat_area_ratio,
            th_color_match_ratio=th_color_match_ratio,
        )
        color_masks.extend(_color_masks)
        palette.extend(_palette)
    return color_masks, palette


def find_flat_color_region(
    rgb_np: np.ndarray,
    mask: np.ndarray,
    max_num_colors: int,
    percentile: float = 0.99,
    th_overlap_ratio: float = 0.5,
    th_flat_area_ratio: float = 0.2,
    th_color_match_ratio: float = 0.85,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Find flat color regions associated with the given mask.

    Args:
        rgb_np: RGB image as numpy array with shape (H, W, 3).
        mask: Binary mask array with shape (H, W) indicating the region of interest.
        max_num_colors: Maximum number of unique colors allowed in the flat region.
            Regions with more colors are filtered out.
        percentile: Percentile threshold for color selection when building the palette.
            Controls how many of the most frequent colors to include. Default: 0.99.
        th_overlap_ratio: Minimum ratio of overlap required between a color region
            and the mask for the region to be included. Default: 0.5.
        th_flat_area_ratio: Minimum ratio of flat (zero gradient) pixels required
            within the masked area. Regions with less flat area are filtered. Default: 0.2.
        th_color_match_ratio: Minimum ratio of pixels in the mask that must match
            the identified palette colors. Default: 0.85.

    Returns:
        A tuple containing:
        - color_masks: List of binary masks (numpy arrays), one for each identified color region.
        - palette: List of RGB color values (numpy arrays) corresponding to each mask.
    """

    assert rgb_np.dtype == np.uint8, f"Expected uint8 rgb_np, got {rgb_np.dtype}"
    assert mask.dtype == bool, f"Expected bool mask, got {mask.dtype}"
    assert rgb_np.ndim == 3 and rgb_np.shape[2] == 3, f"Expected RGB image with shape (H, W, 3), got {rgb_np.shape}"
    assert rgb_np.shape[:2] == mask.shape, f"Shape mismatch: rgb_np {rgb_np.shape[:2]} vs mask {mask.shape}"
    assert 0 < percentile <= 1, f"Percentile must be in (0, 1], got {percentile}"
    assert 0 <= th_overlap_ratio <= 1, f"th_overlap_ratio must be in [0, 1], got {th_overlap_ratio}"
    assert 0 <= th_flat_area_ratio <= 1, f"th_flat_area_ratio must be in [0, 1], got {th_flat_area_ratio}"
    assert 0 <= th_color_match_ratio <= 1, f"th_color_match_ratio must be in [0, 1], got {th_color_match_ratio}"
    assert max_num_colors > 0, f"max_num_colors must be positive, got {max_num_colors}"

    flat_region = calc_gradient_magnitude(rgb_np) == 0
    flat_region_cc = mask & flat_region
    if flat_region_cc.sum() / mask.sum() <= th_flat_area_ratio:  # Filter out non-flat cc
        logger.debug(f"Flat area ratio too small in cc ({flat_region_cc.sum() / mask.sum():.2f}), skipping...")
        return [], []
    colors_in_cc = rgb_np[flat_region_cc]
    color_counts = Counter(map(tuple, colors_in_cc))
    if len(color_counts) > 1000:  # Filter out colorful cc
        logger.debug(f"Too many colors in cc ({len(color_counts)}), skipping...")
        return [], []
    cc_palette = get_colors_until_percentile(color_counts, percentile=percentile)
    if len(cc_palette) > max_num_colors:  # Filter out colorful cc
        logger.debug(f"Too many colors in cc ({len(cc_palette)}), skipping...")
        return [], []

    cc_color_masks = (rgb_np[:, :, None] == cc_palette).all(axis=-1).transpose(2, 0, 1)  # (N, H, W)
    color_match_ratio = np.logical_and(cc_color_masks.any(0), mask).sum() / mask.sum()

    if color_match_ratio < th_color_match_ratio:  # Filter out low color match ratio
        logger.debug(f"Low color match ratio in cc ({color_match_ratio:.2f}), skipping...")
        return [], []

    palette = []
    color_masks = []
    for color, _color_mask in zip(cc_palette, cc_color_masks):
        # Filter out color that does not overlap with cc
        _color_mask_filtered = np.zeros_like(_color_mask)
        for c in divide_mask_to_connected_components(_color_mask):
            if (c & mask).sum() / c.sum() > th_overlap_ratio:
                _color_mask_filtered |= c
        if _color_mask_filtered.sum() > 0:
            color_masks.append(_color_mask_filtered)
            palette.append(color)
    return color_masks, palette


def refine_background(
    bg: np.ndarray, mask: np.ndarray, n_outer_ratio: float, max_num_colors: int, percentile: float = 0.99
) -> np.ndarray:
    """Refine background by assigning closest palette colors to inpainted regions.

    Args:
        bg: Background image as RGB numpy array with shape (H, W, 3), dtype uint8.
        mask: Binary mask array with shape (H, W), dtype bool, indicating
            the inpainted regions to refine.
        n_outer_ratio: Ratio to determine the width of the outer region to sample
            colors from. The actual width is calculated as sqrt(area) * n_outer_ratio.
        max_num_colors: Maximum number of colors to extract for the palette
            from the outer region.
        percentile: Percentile threshold for color frequency when building
            the palette. Default: 0.99.

    Returns:
        Refined background image as numpy array with shape (H, W, 3), dtype uint8,
        where inpainted regions have been replaced with the closest palette colors.
    """

    def _lab_l1_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate L1 distance in LAB color space."""
        a_lab = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
        b_lab = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
        return np.linalg.norm((a_lab - b_lab) * [1.0, 0.5, 0.5], axis=-1)

    assert bg.dtype == np.uint8, f"Expected uint8 bg, got {bg.dtype}"
    assert mask.dtype == bool, f"Expected bool mask, got {mask.dtype}"

    refined_bg = bg.copy()

    mask_ccs = divide_mask_to_connected_components(mask)
    for mask_cc in mask_ccs:
        n_outer = max(int(np.sqrt(mask_cc.sum()) * n_outer_ratio), 5)
        outer_mask = expand_mask(mask_cc, n_outer) & (~mask_cc)
        if outer_mask.sum() == 0:
            continue
        _, palette = find_flat_color_region(
            bg,
            outer_mask,
            max_num_colors=max_num_colors,
            percentile=percentile,
            th_flat_area_ratio=0.0,
            th_overlap_ratio=0.0,
        )
        if len(palette) == 0:
            continue

        # Assign closest palette color to inpainted region
        inpainted_colors = bg[mask_cc]
        distance = _lab_l1_distance(inpainted_colors[:, None, :], np.array(palette)[None, :, :])
        closest_indices = np.argmin(distance, axis=1)
        refined_bg[mask_cc] = np.array(palette)[closest_indices]

    return refined_bg


def estimate_fg_alpha(mask: np.ndarray, fg: np.ndarray, bg: np.ndarray, image: np.ndarray) -> np.ndarray | None:
    """Estimate alpha matte for a given color and background.

    Args:
        mask: Binary mask (H, W) where the color is present
        fg: Foreground color as a numpy array (H, W, 3) or (3,)
        bg: Background image as a numpy array (H, W, 3)
        image: Original image as a numpy array (H, W, 3)

    Returns:
        alpha: Estimated alpha matte (H, W) with values clipped to [0, 1]
    """
    assert mask.dtype == bool, f"Expected bool mask, got {mask.dtype}"
    assert fg.dtype == np.uint8, f"Expected uint8 fg, got {fg.dtype}"
    assert bg.dtype == np.uint8, f"Expected uint8 bg, got {bg.dtype}"

    bg_np = bg.astype(float)
    fg_np = fg.astype(float)
    image_np = image.astype(float)

    mask_to_estimate = mask & ((bg_np != fg_np).any(axis=-1))  # Exclude pixels where bg == fg
    if mask_to_estimate.sum() == 0:
        return None

    num = np.sum((fg_np - bg_np) * (image_np - bg_np), axis=-1)
    denom = np.sum((fg_np - bg_np) ** 2, axis=-1)
    alpha = mask.astype(float)  # Start with binary mask as float
    alpha[mask_to_estimate] = num[mask_to_estimate] / denom[mask_to_estimate]
    alpha = np.clip(alpha, 0, 1)

    return alpha


def estimate_fg_color(
    image_np: np.ndarray,
    bg_np: np.ndarray,
    alpha_np: np.ndarray,
    alpha_clip_range: list[float] = [0, 0.95],
    clip_way: Literal["clip", "replace"] = "clip",
) -> np.ndarray:
    """Estimate foreground RGB from image, background, and alpha (unblending).

    Args:
        image_np: Original image (H, W, 3), uint8
        bg_np: Background image (H, W, 3), uint8
        alpha_np: Alpha mask (H, W), float64 in [0, 1]
        alpha_clip_range: Range for alpha clipping [min, max]
        clip_way: How to handle out-of-range values ("clip" or "replace")

    Returns:
        Estimated foreground RGB (H, W, 3), uint8
    """
    assert image_np.dtype == np.uint8, f"Expected image dtype uint8, got {image_np.dtype}"
    assert bg_np.dtype == np.uint8, f"Expected bg dtype uint8, got {bg_np.dtype}"
    assert alpha_np.dtype == np.float64, f"Expected alpha dtype float64, got {alpha_np.dtype}"
    assert 0 <= alpha_np.min() and alpha_np.max() <= 1, (
        f"Alpha must be in [0, 1], got [{alpha_np.min()}, {alpha_np.max()}]"
    )

    image_float = image_np.astype(float)
    bg_float = bg_np.astype(float)
    alpha_float = alpha_np.copy()

    alpha_float[alpha_float <= alpha_clip_range[0]] = 0
    alpha_float[alpha_float >= alpha_clip_range[1]] = 1
    alpha_float = alpha_float[..., None].repeat(3, axis=-1)

    # Estimate foreground: fg = (image - bg * (1 - alpha)) / alpha
    fg_float = image_float.copy()
    mask = alpha_float > 0
    fg_float[mask] = (image_float[mask] - bg_float[mask] * (1 - alpha_float[mask])) / alpha_float[mask]

    # Handle out-of-range values
    match clip_way:
        case "clip":
            fg_np = np.clip(fg_float, 0, 255).astype(np.uint8)
        case "replace":
            fg_float[(fg_float < 0).any(axis=-1)] = image_float[(fg_float < 0).any(axis=-1)]
            fg_float[(fg_float > 255).any(axis=-1)] = image_float[(fg_float > 255).any(axis=-1)]
            fg_np = np.clip(fg_float, 0, 255).astype(np.uint8)

    return fg_np
