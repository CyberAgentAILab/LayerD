"""Entropy-based element classification."""

from layerd.types import Element

from .base import ElementLabeler
from .utils import compute_entropy


class EntropyLabeler(ElementLabeler):
    """Entropy-based classification for vector/image elements.

    Uses Shannon entropy to distinguish between vector graphics (low entropy)
    and raster images (high entropy). Works well for simple shapes vs photos.

    Args:
        threshold: Entropy threshold (below = vector, above = image). Default: 5.0

    Example:
        >>> from layerd.classification import EntropyLabeler
        >>> labeler = EntropyLabeler(threshold=5.0)
        >>> classified_elements = labeler.classify(elements)
        >>>
        >>> # Check classification results
        >>> for elem in classified_elements:
        ...     print(f"Element {elem['id']}: {elem['type']}")
    """

    def __init__(self, threshold: float = 5.0):
        """Initialize EntropyLabeler.

        Args:
            threshold: Entropy threshold for vector/image separation
        """
        self.threshold = threshold

    def classify(self, elements: list[Element]) -> list[Element]:
        """Classify image elements as vector or image based on entropy.

        Text elements are left unchanged. Image elements are classified
        based on their entropy: low entropy → vector, high entropy → image.

        Args:
            elements: List of Element dicts

        Returns:
            List of Element dicts with updated type field
        """
        result: list[Element] = []
        for elem in elements:
            new_type = elem["type"]
            if elem["type"] == "image":
                entropy = compute_entropy(elem["image"])
                if entropy < self.threshold:
                    new_type = "vector"
            result.append(Element(id=elem["id"], type=new_type, image=elem["image"], box=elem["box"]))
        return result
