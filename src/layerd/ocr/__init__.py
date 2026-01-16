"""OCR module with multiple backend support.

This module provides optical character recognition (OCR) capabilities for the LayerD pipeline.
Two backends are available:

1. **EAST Backend** (Default):
   - Lightweight text detection (~97MB model)
   - CPU and CUDA support
   - Text bounding boxes only (no recognition)
   - No additional dependencies required

2. **Transformers Backend** (GOT-OCR2):
   - Full OCR with text recognition (~1.4GB model)
   - CUDA only (CPU not supported)
   - Requires: pip install layerd[ocr]

Example:
    >>> from layerd.ocr import build_ocr
    >>>
    >>> # EAST backend (default, CPU-compatible)
    >>> ocr = build_ocr("east", device="cpu")
    >>> result = ocr("design.png")
    >>>
    >>> # Transformers backend (CUDA required)
    >>> ocr = build_ocr("transformers", device="cuda")
    >>> result = ocr("design.png")
"""

from typing import TYPE_CHECKING, Any, Literal

from .base import BaseOCR
from .types import BoundingBox, OCRBlock, OCRResult, Polygon

if TYPE_CHECKING:
    from .east_backend import EASTBackend
    from .transformers_backend import TransformersBackend


def build_ocr(backend: Literal["east", "transformers"] = "east", device: str = "cpu", **kwargs: Any) -> BaseOCR:
    """Build OCR backend.

    Factory function following LayerD's pattern (build_matting, build_inpaint).

    Args:
        backend: Backend type
            - "east": EAST text detector (CPU/CUDA, lightweight, no recognition)
            - "transformers": GOT-OCR2 (CUDA only, full OCR with recognition)
        device: Device for inference ("cpu", "cuda", "cuda:0", etc.)
        **kwargs: Backend-specific parameters

    Returns:
        Initialized OCR backend

    Raises:
        ValueError: If backend not supported or invalid device for backend
        ImportError: If optional dependencies not installed (transformers backend)

    Examples:
        >>> # EAST backend (default, lightweight, CPU-compatible)
        >>> ocr = build_ocr("east", device="cpu")
        >>> result = ocr(image)
        >>> print(f"Found {len(result['blocks'])} text regions")

        >>> # Transformers backend (full OCR, CUDA required)
        >>> ocr = build_ocr("transformers", device="cuda")
        >>> result = ocr(image)
        >>> for block in result['blocks']:
        ...     print(f"Text: {block['text']}")

        >>> # Batch processing
        >>> results = ocr([image1, image2, image3])

        >>> # Device switching
        >>> ocr = ocr.to("cuda:1")
    """
    # Lazy import backends only when needed
    if backend == "east":
        from .east_backend import EASTBackend  # noqa: PLC0415

        return EASTBackend(device=device, **kwargs)
    elif backend == "transformers":
        from .transformers_backend import TransformersBackend  # noqa: PLC0415

        return TransformersBackend(device=device, **kwargs)
    else:
        raise ValueError(
            f"Backend '{backend}' not supported. "
            f"Available backends: 'east' (CPU/CUDA), 'transformers' (CUDA only)"
        )


__all__ = [
    # Base class
    "BaseOCR",
    # Factory function
    "build_ocr",
    # Types
    "BoundingBox",
    "OCRResult",
    "OCRBlock",
    "Polygon",
]
