"""Inpaint models and registry."""

from typing import Any

from layerd.models.inpaint.base import BaseInpaint
from layerd.models.inpaint.lama_inpaint import LamaInpaint

INPAINT_MODELS = {
    "lama": LamaInpaint,
}


def build_inpaint(name: str, **kwargs: Any) -> BaseInpaint:
    """Create an inpaint model by name.

    Args:
        name: Name of the inpaint model (e.g., "lama")
        **kwargs: Additional arguments for model initialization

    Returns:
        Initialized inpaint model instance

    Raises:
        ValueError: If the model name is not found in registry
    """
    if name not in INPAINT_MODELS:
        available = list(INPAINT_MODELS.keys())
        raise ValueError(f"Unknown inpaint model: {name}. Available: {available}")
    return INPAINT_MODELS[name](**kwargs)


__all__ = ["BaseInpaint", "build_inpaint", "LamaInpaint"]
