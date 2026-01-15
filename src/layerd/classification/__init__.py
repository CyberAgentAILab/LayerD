"""Element classification module for LayerD.

Provides classifiers for distinguishing between element types:
- Vector graphics (simple shapes with low complexity)
- Raster images (photos, complex artwork)
- Text (from OCR integration)
"""

from layerd.classification.base import ElementLabeler
from layerd.classification.entropy import EntropyLabeler
from layerd.classification.gradient import GradientAwareLabeler
from layerd.classification.utils import compute_entropy

__all__ = [
    "ElementLabeler",
    "EntropyLabeler",
    "GradientAwareLabeler",
    "compute_entropy",
]
