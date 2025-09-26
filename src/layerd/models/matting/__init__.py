"""Matting models and registry."""

from typing import Any

from layerd.models.matting.base import BaseMatting
from layerd.models.matting.birefnet_matting import BiRefNetMatting

MATTING_MODELS = {
    "birefnet": BiRefNetMatting,
}


def build_matting(name: str, **kwargs: Any) -> BaseMatting:
    """Create a matting model by name.

    Args:
        name: Name of the matting model (e.g., "birefnet")
        **kwargs: Additional arguments for model initialization

    Returns:
        Initialized matting model instance

    Raises:
        ValueError: If the model name is not found in registry
    """
    if name not in MATTING_MODELS:
        available = list(MATTING_MODELS.keys())
        raise ValueError(f"Unknown matting model: {name}. Available: {available}")
    return MATTING_MODELS[name](**kwargs)


__all__ = ["BaseMatting", "build_matting", "BiRefNetMatting"]
