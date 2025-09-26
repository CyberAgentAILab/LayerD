from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class BaseMatting(ABC):
    """Base class for matting models for validating inputs and outputs."""

    def __init__(self) -> None:
        pass

    def __call__(self, image: Image.Image | np.ndarray) -> np.ndarray:
        self._validate_inputs(image)
        alpha = self.infer(image)
        self._validate_outputs(alpha, image)
        return alpha

    @abstractmethod
    def infer(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Infer the alpha matte from the input image."""
        pass

    def _validate_inputs(self, image: Image.Image | np.ndarray) -> None:
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                raise ValueError(f"Image mode should be RGB, but got {image.mode}")
        elif isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(f"Expected 3D array (H, W, 3), got shape {image.shape}")
            if image.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got {image.shape[2]}")
            if image.dtype != np.uint8:
                raise ValueError(f"Expected uint8 dtype, got {image.dtype}")
        else:
            raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(image)}")

    def _validate_outputs(self, alpha: np.ndarray, image: Image.Image | np.ndarray) -> None:
        if not isinstance(alpha, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(alpha)}")

        # Check dtype is float64
        if alpha.dtype != np.float64:
            raise ValueError(f"Expected float64 dtype, got {alpha.dtype}")

        # Check value range
        if alpha.min() < 0 or alpha.max() > 1:
            raise ValueError(f"Alpha values must be in [0, 1], got [{alpha.min()}, {alpha.max()}]")

        # Get expected dimensions based on input type
        if isinstance(image, Image.Image):
            expected_height, expected_width = image.size[::-1]  # PIL uses (width, height)
        elif isinstance(image, np.ndarray):
            expected_height, expected_width = image.shape[:2]  # numpy uses (height, width)
        else:
            raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(image)}")

        if alpha.shape[:2] != (expected_height, expected_width):
            raise ValueError(
                f"Output shape {alpha.shape[:2]} does not match input image size {(expected_height, expected_width)}"
            )

    @abstractmethod
    def to(self, device: str) -> "BaseMatting":
        """Move model to specified device (e.g., 'cpu' or 'cuda')."""
        pass
