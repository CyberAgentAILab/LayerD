"""Export module for LayerD elements to various formats."""

from typing import Any, Literal, overload

from .base import BaseExporter
from .psd import PSDBuilder
from .svg import SVGBuilder, SVGParser

# Lazy registry
_EXPORTER_REGISTRY: dict[str, type[BaseExporter[Any]]] = {}


def _populate_registry() -> None:
    """Populate exporter registry with available formats."""
    if not _EXPORTER_REGISTRY:
        _EXPORTER_REGISTRY["svg"] = SVGBuilder
        _EXPORTER_REGISTRY["psd"] = PSDBuilder


@overload
def build_exporter(format_type: Literal["svg"] = "svg", **kwargs: Any) -> BaseExporter[str]: ...


@overload
def build_exporter(format_type: Literal["psd"], **kwargs: Any) -> BaseExporter[bytes]: ...


def build_exporter(format_type: Literal["svg", "psd"] = "svg", **kwargs: Any) -> BaseExporter[Any]:
    """Build format exporter.

    Factory function following LayerD's pattern (build_matting, build_ocr).

    Args:
        format_type: Export format ("svg", "psd")
        **kwargs: Format-specific configuration

    Returns:
        Initialized exporter

    Raises:
        ValueError: If format not available

    Examples:
        >>> # SVG with base64 images
        >>> exporter = build_exporter("svg")
        >>> svg = exporter(elements, canvas_size)
        >>>
        >>> # SVG with external images
        >>> exporter = build_exporter("svg", image_mode="external", image_dir="./images")
        >>> svg = exporter(elements, canvas_size)
        >>>
        >>> # PSD export
        >>> exporter = build_exporter("psd")
        >>> psd_bytes = exporter(elements, canvas_size)
    """
    # Lazy load registry
    _populate_registry()

    if format_type not in _EXPORTER_REGISTRY:
        available = list(_EXPORTER_REGISTRY.keys())
        error_msg = f"Format '{format_type}' not available."

        if available:
            error_msg += f" Available formats: {available}."

        raise ValueError(error_msg)

    exporter_cls = _EXPORTER_REGISTRY[format_type]
    return exporter_cls(**kwargs)


__all__ = [
    "BaseExporter",
    "SVGBuilder",
    "SVGParser",
    "PSDBuilder",
    "build_exporter",
]
