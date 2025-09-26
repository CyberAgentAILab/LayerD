import random

import numpy as np
from PIL import Image, ImageEnhance


def preproc(
    image: Image.Image, label: Image.Image, preproc_methods: list[str] = ["flip"]
) -> tuple[Image.Image, Image.Image]:
    if "flip" in preproc_methods:
        image, label = cv_random_flip(image, label)
    if "rotate" in preproc_methods:
        image, label = random_rotate(image, label)
    if "enhance" in preproc_methods:
        image = color_enhance(image)
    if "pepper" in preproc_methods:
        image = random_pepper(image)
    return image, label


def cv_random_flip(img: Image.Image, label: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() > 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img, label


def random_rotate(image: Image.Image, label: Image.Image, angle: int = 15) -> tuple[Image.Image, Image.Image]:
    mode = Image.Resampling.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-angle, angle)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def color_enhance(image: Image.Image) -> Image.Image:
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def random_pepper(img: Image.Image, N: float = 0.0015) -> Image.Image:
    img_array = np.array(img)
    noiseNum = int(N * img_array.shape[0] * img_array.shape[1])
    for _ in range(noiseNum):
        randX = random.randint(0, img_array.shape[0] - 1)
        randY = random.randint(0, img_array.shape[1] - 1)
        img_array[randX, randY] = random.randint(0, 1) * 255
    return Image.fromarray(img_array)
