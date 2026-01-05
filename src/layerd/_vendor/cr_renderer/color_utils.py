import logging
from typing import Tuple

import skia  # type: ignore

logger = logging.getLogger(__name__)


def parse_color(name: str, opacity: float = 1.0) -> int:
    """Get 24-bit color integer for skia."""

    # TODO: Consider using a full CSS color parser.

    def clip_uint8(value: float) -> int:
        return int(max(min(value, 255), 0))

    # From color component.
    if name.startswith("rgba("):
        rgba = tuple(map(float, name.lstrip("rgba(").rstrip(")").split(",")))
        return skia.ColorSetARGB(
            a=clip_uint8(rgba[3] * opacity * 255),
            r=clip_uint8(rgba[0]),
            g=clip_uint8(rgba[1]),
            b=clip_uint8(rgba[2]),
        )
    elif name.startswith("rgb("):
        rgb = tuple(map(float, name.lstrip("rgb(").rstrip(")").split(",")))
        return skia.ColorSetARGB(
            a=clip_uint8(opacity * 255),
            r=clip_uint8(rgb[0]),
            g=clip_uint8(rgb[1]),
            b=clip_uint8(rgb[2]),
        )
    elif name.startswith("cmyka("):
        cmyka = tuple(map(float, name.lstrip("cmyka(").rstrip(")").split(",")))
        rgb = cmyk_to_rgb(cmyka[0], cmyka[1], cmyka[2], cmyka[3])
        return skia.ColorSetARGB(
            a=clip_uint8(cmyka[4] * opacity * 255),
            r=clip_uint8(rgb[0]),
            g=clip_uint8(rgb[1]),
            b=clip_uint8(rgb[2]),
        )
    elif name.startswith("cmyk("):
        cmyk = tuple(map(float, name.lstrip("cmyka(").rstrip(")").split(",")))
        rgb = cmyk_to_rgb(cmyk[0], cmyk[1], cmyk[2], cmyk[3])
        return skia.ColorSetARGB(
            a=clip_uint8(opacity * 255),
            r=clip_uint8(rgb[0]),
            g=clip_uint8(rgb[1]),
            b=clip_uint8(rgb[2]),
        )
    elif name.startswith("#") or len(name) == 3 or len(name) == 6:
        # From HEX code.
        hex = name.lstrip("#").replace("none", "000000").replace("undefined", "000000")
        if len(hex) == 3:
            hex = "".join([c * 2 for c in hex])
        if len(hex) != 6:
            raise ValueError(f"Invalid color HEX code: {name}")
        return skia.ColorSetA(c=int(hex, base=16), a=int(opacity * 255))

    raise NotImplementedError(f"Unknown color format: {name}")


def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> Tuple[int, int, int]:
    """Convert CMYK to RGB."""
    r = int(255 * (100 - c) / 100.0 * (100 - k) / 100.0)
    g = int(255 * (100 - m) / 100.0 * (100 - k) / 100.0)
    b = int(255 * (100 - y) / 100.0 * (100 - k) / 100.0)
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255
    return r, g, b


def to_hex(color: int) -> str:
    """Convert skia color to HEX code."""
    return "#" + ("%08x" % color)[2:]


def to_rgba(color: int) -> str:
    """Convert skia color to RGBA components."""
    return "rgba(%d, %d, %d, %g)" % (
        skia.ColorGetR(color),
        skia.ColorGetG(color),
        skia.ColorGetB(color),
        (skia.ColorGetA(color) / 255.0),
    )


def parse_to_rgba(name: str, opacity: float = 1.0) -> str:
    """Parse color name and convert to RGBA."""
    return to_rgba(parse_color(name, opacity))
