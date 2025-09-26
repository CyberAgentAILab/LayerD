import logging
import os
from glob import glob

import cv2
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from .image_proc import preproc

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # remove DecompressionBombWarning


def path_to_image(
    path: str, size: tuple[int, int] | list[int] = (1024, 1024), color_type: str = "rgb"
) -> Image.Image | None:
    if color_type.lower() == "rgb":
        image = cv2.imread(path)
    elif color_type.lower() == "gray":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        logger.warning("Select the color_type to return, either to RGB or gray image.")
        return None
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == "rgb":
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("L")
    return pil_image


class MattingDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int | list[int],
        is_train: bool = True,
        preproc_methods: list[str] | None = None,
    ) -> None:
        if preproc_methods is None:
            preproc_methods = []
        # Handle int image_size as square
        if isinstance(image_size, int):
            self.image_size = [image_size, image_size]
        else:
            self.image_size = image_size
        self.is_train = is_train
        self.preproc_methods = preproc_methods
        valid_extensions = [".png", ".jpg", ".PNG", ".JPG", ".JPEG"]

        self.transform_image = transforms.Compose(
            [
                transforms.Resize(self.image_size[::-1]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform_label = transforms.Compose(
            [
                transforms.Resize(self.image_size[::-1]),
                transforms.ToTensor(),
            ]
        )
        dataset_root = root
        self.image_paths = glob(os.path.join(dataset_root, split, "im", "*.png"))
        self.label_paths = []
        for p in self.image_paths:
            for ext in valid_extensions:
                p_gt = p.replace("/im/", "/gt/")[: -(len(p.split(".")[-1]) + 1)] + ext
                file_exists = False
                if os.path.exists(p_gt):
                    self.label_paths.append(p_gt)
                    file_exists = True
                    break
            if not file_exists:
                logger.warning(f"Label file not found for image: {p}")

        if len(self.label_paths) != len(self.image_paths):
            raise ValueError(
                f"There are different numbers of images ({len(self.label_paths)}) and labels ({len(self.image_paths)})"
            )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = path_to_image(self.image_paths[index], size=self.image_size, color_type="rgb")
        label = path_to_image(self.label_paths[index], size=self.image_size, color_type="gray")

        if image is None or label is None:
            raise ValueError(f"Failed to load image or label at index {index}")

        # loading image and label
        if self.is_train:
            image, label = preproc(image, label, preproc_methods=self.preproc_methods)
        image, label = self.transform_image(image), self.transform_label(label)

        return image, label

    def __len__(self) -> int:
        return len(self.image_paths)
