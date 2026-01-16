"""Type definitions for OCR module.

This module defines the core types used throughout the OCR system:
- BoundingBox: Rectangular bounding box coordinates
- Polygon: Arbitrary polygon vertex coordinates
- OCRBlock: Single text block with bounding box and optional metadata
- OCRResult: Complete OCR result for an image

All types use TypedDict for JSON serialization compatibility.
"""

from typing import Any, NotRequired, TypedDict


class BoundingBox(TypedDict):
    """Bounding box coordinates.

    Represents a rectangular region in image coordinates.
    Origin (0,0) is at the top-left corner of the image.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Polygon(TypedDict):
    """Polygon vertex coordinates."""

    x: int
    y: int


class OCRBlock(TypedDict):
    """Single text block with bounding box.

    Core fields:
        text: Recognized text content
        bbox: Rectangular bounding box

    Optional fields:
        confidence: Recognition confidence score (0.0-1.0)
        polygon: Arbitrary polygon vertices (fallback to bbox if not needed)
        block_type: Semantic classification ("text", "title", "table", etc.)
    """

    text: str
    bbox: BoundingBox

    # Optional fields
    confidence: NotRequired[float]
    polygon: NotRequired[list[Polygon]]
    block_type: NotRequired[str]


class OCRResult(TypedDict):
    """OCR result for a single image.

    Minimal schema: (image_size, list of (text, bbox))

    Core fields:
        image_size: Original image dimensions (width, height)
        blocks: List of detected text blocks

    Optional fields:
        metadata: Additional information (model name, backend, processing time, etc.)
    """

    image_size: tuple[int, int]
    blocks: list[OCRBlock]

    # Optional metadata
    metadata: NotRequired[dict[str, Any]]


__all__ = ["BoundingBox", "Polygon", "OCRBlock", "OCRResult"]
