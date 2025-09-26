import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama

from layerd.models.inpaint.base import BaseInpaint


class LamaInpaint(BaseInpaint):
    """LaMa inpainting model wrapper following the BaseInpaint interface."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.model = SimpleLama(device)

    def infer(self, image_np: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
        """Perform inpainting using LaMa model.

        Args:
            image_np: Input image (RGB)
            hard_mask: Boolean numpy mask (False=keep, True=inpaint)

        Returns:
            Inpainted image as numpy array
        """
        # Convert boolean mask to uint8 (0 or 255) as required by SimpleLama
        mask_np = hard_mask.astype(np.uint8) * 255

        # Store original dimensions
        original_height, original_width = image_np.shape[:2]

        # Run inpainting
        result_np = np.array(self.model(image_np, mask_np))[:original_height, :original_width]

        return result_np

    def to(self, device: str) -> "LamaInpaint":
        """Move model to specified device (e.g., 'cpu' or 'cuda')."""
        self.model.model.to(device)
        self.model.device = device
        return self
