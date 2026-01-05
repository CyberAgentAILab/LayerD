import io
import logging

import PIL.Image
import skia  # type: ignore

logger = logging.getLogger(__name__)


def encode_surface(surface: skia.Surface, format: str) -> bytes:
    """Convert skia.Surface to bytes."""
    return encode_skia_image(surface.makeImageSnapshot(), format)


def encode_skia_image(image: skia.Image, format: str) -> bytes:
    """Convert skia.Image to bytes."""
    formats = dict(png=skia.kPNG, jpeg=skia.kJPEG)
    with io.BytesIO() as f:
        image.save(f, formats[format])
        return f.getvalue()


def encode_pil_image(image: PIL.Image.Image, format: str) -> bytes:
    """Convert PIL.Image to bytes."""
    with io.BytesIO() as f:
        image.save(f, format)
        return f.getvalue()


def decode_skia_image(image_bytes: bytes) -> skia.Image:
    """Convert bytes to skia.Image."""
    with io.BytesIO(image_bytes) as f:
        return skia.Image.open(f)


def convert_pil_image_to_skia_image(
    image: PIL.Image.Image, format: str = "PNG"
) -> skia.Image:
    """Convert PIL.Image to skia.Image."""
    image_bytes = encode_pil_image(image, format=format)
    return decode_skia_image(image_bytes)
