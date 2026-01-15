"""Base class for format exporters."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from layerd.types import Element

# Type variable for output format (str for SVG, bytes for PSD)
OutputT = TypeVar("OutputT", str, bytes)


class BaseExporter(ABC, Generic[OutputT]):
    """Abstract base class for layer export formats.

    Follows the ElementLabeler pattern with strategy-based configuration.
    Subclasses implement specific format export logic (SVG, PSD, etc.).

    Type Parameters:
        OutputT: Output format type (str for text formats, bytes for binary)

    Example:
        >>> class SVGBuilder(BaseExporter[str]):
        ...     def export(self, elements, canvas_size):
        ...         return "<svg>...</svg>"
    """

    def __init__(self, **config: Any) -> None:
        """Initialize exporter with format-specific configuration.

        The base implementation accepts configuration options but does not use them,
        allowing subclasses flexibility in their initialization signatures.

        Subclasses should call ``super().__init__()`` before their own initialization
        to maintain consistency with the base class interface.

        Args:
            **config: Format-specific configuration options (interpreted by subclasses)
        """
        pass

    @abstractmethod
    def export(self, elements: list[Element], canvas_size: tuple[int, int]) -> OutputT:
        """Export elements to format-specific representation.

        Args:
            elements: List of processed elements with images and bounding boxes
            canvas_size: Canvas dimensions as (width, height)

        Returns:
            Format-specific output (str for text formats, bytes for binary)
        """
        raise NotImplementedError

    def __call__(self, elements: list[Element], canvas_size: tuple[int, int]) -> OutputT:
        """Convenience method for export.

        Enables: exporter(elements, canvas_size)

        Args:
            elements: List of processed elements
            canvas_size: Canvas dimensions

        Returns:
            Format-specific output
        """
        return self.export(elements, canvas_size)
