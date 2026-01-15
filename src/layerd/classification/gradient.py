"""Gradient-aware element classification."""

import numpy as np
from PIL import Image

from layerd.types import Element

from .base import ElementLabeler
from .utils import compute_entropy


class GradientAwareLabeler(ElementLabeler):
    """Classifier that detects gradients and handles them appropriately.

    Addresses the limitation of pure entropy-based classification which can
    misclassify gradients. Gradients have low entropy but should remain as
    raster images since they can't be easily vectorized.

    Args:
        entropy_threshold: Entropy threshold for vector/image classification (default: 5.0)
        gradient_threshold: Pixel difference threshold for gradient detection (default: 0.3)
                          Higher values = less sensitive to gradients

    Example:
        >>> from layerd.classification import GradientAwareLabeler
        >>> labeler = GradientAwareLabeler(
        ...     entropy_threshold=5.0,
        ...     gradient_threshold=0.3
        ... )
        >>> classified_elements = labeler.classify(elements)
    """

    def __init__(
        self,
        entropy_threshold: float = 5.0,
        gradient_threshold: float = 0.3,
    ):
        """Initialize GradientAwareLabeler.

        Args:
            entropy_threshold: Entropy threshold for vector/image classification
            gradient_threshold: Average pixel difference threshold for gradient detection.
                              Mean absolute difference between adjacent pixels (0-255 scale).
                              Typical range: 0.1-1.0
        """
        self.entropy_threshold = entropy_threshold
        self.gradient_threshold = gradient_threshold

    def classify(self, elements: list[Element]) -> list[Element]:
        """Classify elements with gradient detection.

        Classification logic:
        1. Text elements → unchanged
        2. If gradient detected → keep as "image" (raster)
        3. Else if low entropy → "vector"
        4. Else → "image"

        Args:
            elements: List of Element dicts

        Returns:
            List of Element dicts with updated type field
        """
        result: list[Element] = []
        for elem in elements:
            new_type = elem["type"]
            if elem["type"] == "image":
                image = elem["image"]

                # Gradient detection first (overrides entropy)
                if self._has_gradient(image):
                    new_type = "image"  # Keep as raster
                elif compute_entropy(image) < self.entropy_threshold:
                    new_type = "vector"
                else:
                    new_type = "image"

            result.append(Element(id=elem["id"], type=new_type, image=elem["image"], box=elem["box"]))
        return result

    def _has_gradient(self, image: Image.Image) -> bool:
        """Detect smooth gradients using pixel difference analysis.

        Gradients exhibit consistent but non-zero color changes between
        adjacent pixels. This heuristic checks for that pattern.

        Args:
            image: PIL Image (RGBA)

        Returns:
            True if gradient detected, False otherwise
        """
        arr = np.array(image.convert("RGB"))

        # Check minimum size
        h, w = arr.shape[:2]
        if h < 10 or w < 10:
            return False

        # Sample horizontal and vertical gradients
        h_grad = np.abs(np.diff(arr, axis=1)).mean()
        v_grad = np.abs(np.diff(arr, axis=0)).mean()

        # Average gradient strength
        avg_grad = (h_grad + v_grad) / 2

        return avg_grad > self.gradient_threshold
