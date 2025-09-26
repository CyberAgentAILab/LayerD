import logging
from typing import cast

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from layerd.matting.birefnet import build_birefnet
from layerd.models.matting.base import BaseMatting

logger = logging.getLogger(__name__)


class BiRefNetMatting(BaseMatting):
    """BiRefNet wrapper for matting operations following the BaseMatting interface."""

    def __init__(
        self,
        hf_card: str = "cyberagent/layerd-birefnet",
        process_image_size: tuple[int, int] | None = None,
        device: str = "cpu",
        weight_path: str | None = None,
    ) -> None:
        super().__init__()
        self.model = build_birefnet(hf_card)
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from {weight_path}")
        self.model.to(device)
        self.model.eval()
        self.device = device

        # Use model's trained size if available and not overridden
        if process_image_size is None:
            if hasattr(self.model, "config") and hasattr(self.model.config, "size"):
                default_size = cast(int, self.model.config.size)
                self.process_image_size = (default_size, default_size)
                logger.info(f"Using model's trained size: {self.process_image_size}, as no size was specified")
            else:
                self.process_image_size = (512, 512)
                logger.warning("Could not get model's trained size, using default: (512, 512)")
        else:
            self.process_image_size = process_image_size
            # Warn if different from model's trained size
            if hasattr(self.model, "config") and hasattr(self.model.config, "size"):
                default_size = cast(int, self.model.config.size)
                if process_image_size[0] != default_size or process_image_size[1] != default_size:
                    logger.warning(
                        f"Using size {process_image_size} which differs from model's trained size ({default_size}, {default_size})"
                    )

    def infer(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            h, w = image.shape[:2]
        else:
            pil_image = image
            w, h = pil_image.size

        transform = transforms.Compose(
            [
                transforms.Resize(self.process_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet mean/std
            ]
        )
        input_tensor = transform(pil_image).unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            preds = self.model(input_tensor)[0][-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred = cv2.resize(pred.numpy(), (w, h), interpolation=cv2.INTER_LINEAR)

        return pred.astype(np.float64)

    def to(self, device: str) -> "BiRefNetMatting":
        """Move model to specified device (e.g., 'cpu' or 'cuda')."""
        self.model.to(device)
        self.device = device
        return self
