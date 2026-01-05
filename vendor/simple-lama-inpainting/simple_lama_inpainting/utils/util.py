import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse


# Source https://github.com/advimman/lama
def get_image(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img: np.ndarray, factor: float, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(
    image: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    device: torch.device,
    pad_out_to_modulo: int = 8,
    scale_factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_image_np = get_image(image)
    out_mask_np = get_image(mask)

    if scale_factor is not None:
        out_image_np = scale_image(out_image_np, scale_factor)
        out_mask_np = scale_image(out_mask_np, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image_np = pad_img_to_modulo(out_image_np, pad_out_to_modulo)
        out_mask_np = pad_img_to_modulo(out_mask_np, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image_np).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask_np).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


# Source: https://github.com/Sanster/lama-cleaner/blob/6cfc7c30f1d6428c02e21d153048381923498cac/lama_cleaner/helper.py # noqa
def get_cache_path_by_url(url: str) -> str:
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url: str) -> str:
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
    return cached_file
