"""Utility functions for element classification."""

import numpy as np
from PIL import Image


def compute_entropy(image: Image.Image) -> float:
    """Compute alpha-weighted Shannon entropy of image pixel distribution.

    Higher entropy indicates more complex/photographic content.
    Lower entropy indicates simpler/vector-like content.

    Args:
        image: PIL Image (RGBA)

    Returns:
        Entropy value (higher = more complex, lower = simpler)

    Example:
        >>> from PIL import Image
        >>> from layerd.classification.utils import compute_entropy
        >>> image = Image.open("design_element.png")
        >>> entropy = compute_entropy(image)
        >>> if entropy < 5.0:
        ...     print("Likely vector graphics")
        ... else:
        ...     print("Likely raster image")
    """
    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3].flatten().astype(np.float64) / 255.0
    rgb = arr[:, :, :3].reshape(-1, 3)

    # Skip if no visible pixels
    if alpha.sum() < 1e-6:
        return 0.0

    # Quantize colors (256 -> 32 levels per channel)
    quantized = (rgb // 8).astype(np.uint32)
    color_indices = quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]

    # Alpha-weighted histogram
    unique_colors, inverse = np.unique(color_indices, return_inverse=True)
    weights = np.zeros(len(unique_colors), dtype=np.float64)
    np.add.at(weights, inverse, alpha)

    # Normalize to probabilities
    total_weight = weights.sum()
    if total_weight < 1e-6:
        return 0.0
    probs = weights / total_weight

    # Shannon entropy
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)
