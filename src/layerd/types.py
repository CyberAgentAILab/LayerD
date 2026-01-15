"""Common type definitions for LayerD.

This module provides shared type definitions used across LayerD's postprocessing
and export features.

Example:
    >>> from layerd.types import BoundingBox, Element
    >>> from PIL import Image
    >>>
    >>> # Create a bounding box
    >>> bbox: BoundingBox = {
    ...     "x_min": 10,
    ...     "y_min": 20,
    ...     "x_max": 100,
    ...     "y_max": 80
    ... }
    >>>
    >>> # Create an element
    >>> element: Element = {
    ...     "id": 1,
    ...     "type": "text",
    ...     "image": Image.new("RGBA", (90, 60)),
    ...     "box": bbox
    ... }
"""

from typing import TypedDict

from PIL import Image


class BoundingBox(TypedDict):
    """Bounding box coordinates.

    Represents a rectangular region with inclusive lower bounds and exclusive upper bounds.
    Used by OCR, postprocessing, and export modules.

    Attributes:
        x_min: Left edge (inclusive)
        y_min: Top edge (inclusive)
        x_max: Right edge (exclusive)
        y_max: Bottom edge (exclusive)

    Example:
        >>> bbox: BoundingBox = {
        ...     "x_min": 0,
        ...     "y_min": 0,
        ...     "x_max": 100,
        ...     "y_max": 50
        ... }
        >>> width = bbox["x_max"] - bbox["x_min"]  # 100
        >>> height = bbox["y_max"] - bbox["y_min"]  # 50
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Element(TypedDict):
    """Processed element output format.

    Represents a single design element extracted from layer decomposition,
    with type classification and spatial information.

    Attributes:
        id: Unique element identifier
        type: Element type - one of "text", "vector", or "image"
        image: Cropped RGBA image of the element
        box: Bounding box coordinates relative to original canvas

    Example:
        >>> from PIL import Image
        >>> element: Element = {
        ...     "id": 1,
        ...     "type": "text",
        ...     "image": Image.new("RGBA", (100, 50)),
        ...     "box": {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 70}
        ... }
    """

    id: int
    type: str  # "text", "vector", or "image"
    image: Image.Image  # Cropped RGBA image
    box: BoundingBox


__all__ = ["BoundingBox", "Element"]
