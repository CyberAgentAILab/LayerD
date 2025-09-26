from abc import ABC, abstractmethod

import numpy as np


class BaseInpaint(ABC):
    """Abstract base class for inpainting models."""

    def __init__(self) -> None:
        pass

    def __call__(self, image: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
        """Inpaint the masked regions of an image.

        Args:
            image: Input image (RGB)
            hard_mask: Boolean numpy mask (False=keep, True=inpaint)

        Returns:
            Inpainted image as numpy array
        """
        self._validate_inputs(image, hard_mask)
        result = self.infer(image, hard_mask)
        self._validate_outputs(result, image)
        return result

    @abstractmethod
    def infer(self, image: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
        """Perform the actual inpainting.

        Args:
            image: Input image (RGB)
            hard_mask: Boolean numpy mask (False=keep, True=inpaint)

        Returns:
            Inpainted image as numpy array
        """
        pass

    def _validate_inputs(self, image: np.ndarray, hard_mask: np.ndarray) -> None:
        """Validate input image and hard mask."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be np.ndarray, got {type(image)}")

        if not isinstance(hard_mask, np.ndarray):
            raise TypeError(f"Hard mask must be np.ndarray, got {type(hard_mask)}")

        # Check that mask is boolean dtype
        if hard_mask.dtype != np.bool_:
            raise TypeError(f"Hard mask must be boolean dtype (np.bool_), got {hard_mask.dtype}")

        # Check dimensions match
        img_w, img_h = image.shape[1], image.shape[0]
        mask_h, mask_w = hard_mask.shape[:2]

        if (img_h, img_w) != (mask_h, mask_w):
            raise ValueError(f"Image size ({img_h}, {img_w}) does not match hard mask size ({mask_h}, {mask_w})")

    def _validate_outputs(self, result: np.ndarray, original_image: np.ndarray) -> None:
        """Validate output matches input format."""
        if not isinstance(result, np.ndarray):
            raise TypeError(f"Output must be np.ndarray, got {type(result)}")

        if result.shape[:2] != original_image.shape[:2]:
            raise ValueError(f"Output size {result.shape[:2]} does not match input size {original_image.shape[:2]}")

    @abstractmethod
    def to(self, device: str) -> "BaseInpaint":
        """Move model to specified device (e.g., 'cpu' or 'cuda')."""
        pass
