import logging
import math
import re
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple

import datasets  # type: ignore
import skia  # type: ignore

import cr_renderer.image_utils as image_utils
import cr_renderer.text_utils as text_utils
from cr_renderer.fonts import FontManager
from cr_renderer.schema import TextElement

logger = logging.getLogger(__name__)


class _BaseRenderer(object):
    """Base renderer."""

    def __init__(self, features: datasets.Features, fonts_path: Optional[str] = None):
        self.features = features
        self.font_manager = FontManager(fonts_path)

    def render(
        self,
        example: Dict[str, Any],
        max_size: int = 360,
        render_text: bool = True,
    ) -> bytes:
        """Render a preprocessed example and return as JPEG bytes."""
        raise NotImplementedError("Subclasses must implement this method.")


class CrelloV5Renderer(_BaseRenderer):
    """Render an example for Crello v5 dataset.

    Example:

        import datasets
        from cr_renderer import CrelloV5Renderer

        dataset = datasets.load_dataset(
            "cyberagent/crello", revision="5.0.0", split="train")
        renderer = CrelloV5Renderer(dataset.features)
        for example in dataset:
            image_bytes = renderer.render(example)
    """

    def render(
        self,
        example: Dict[str, Any],
        max_size: int = 360,
        render_text: bool = True,
        format: str = "jpeg",
        canvas_color: skia.Color4f | None = skia.ColorWHITE
    ) -> bytes:
        """Render a preprocessed example and return as JPEG bytes."""
        example = _decode_class_label(self.features, example)
        # TODO: validate the example against the pydantic schema.
        return _render_to_surface(self.font_manager, example, max_size, render_text, format, canvas_color)


class CrelloV4Renderer(_BaseRenderer):
    """Render an example for Crello v4 dataset.

    Example:

        import datasets
        from cr_renderer import CrelloV4Renderer

        dataset = datasets.load_dataset(
            "cyberagent/crello", revision="4.0.0", split="train")
        renderer = CrelloV4Renderer(dataset.features)
        for example in dataset:
            image_bytes = renderer.render(example)
    """

    def render(
        self,
        example: Dict[str, Any],
        max_size: int = 360,
        render_text: bool = True,
        format: str = "jpeg",
        canvas_color: skia.Color4f | None = skia.ColorWHITE
    ) -> bytes:
        """Render a preprocessed example and return as JPEG bytes."""
        example = _decode_class_label(self.features, example)
        example = self.convert_to_v5(example)
        return _render_to_surface(self.font_manager, example, max_size, render_text, format, canvas_color)

    @staticmethod
    def convert_to_v5(example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Crello v4 example to Crello v5 format."""
        length = int(example["length"])
        type_name = [
            example["type"][i][:1].capitalize() + example["type"][i][1:]
            for i in range(length)
        ]
        canvas_width = int(example["canvas_width"])
        canvas_height = int(example["canvas_height"])

        text = [re.sub("\n+", "\n", x) for x in example["text"]]

        text_color = [
            [
                "rgba({},{},{},{})".format(
                    *example["color"][i], float(example["opacity"][i])
                )
            ]
            * len(text[i])
            for i in range(length)
        ]
        text_line = [
            list(accumulate([int(c == "\n") for c in list(text[i])]))
            for i in range(length)
        ]

        return {
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "length": int(example["length"]),
            "left": [(x * canvas_width) for x in example["left"]],
            "top": [(x * canvas_height) for x in example["top"]],
            "width": [(x * canvas_width) for x in example["width"]],
            "height": [(x * canvas_height) for x in example["height"]],
            "angle": [(180.0 * x / math.pi) for x in example["angle"]],
            "type": type_name,
            "color": [],
            "opacity": example["opacity"],
            "text": text,
            "font_size": example["font_size"],
            "font": example["font"],
            "line_height": example["line_height"],
            "text_align": example["text_align"],
            "capitalize": [x == "true" for x in example["capitalize"]],
            "letter_spacing": example["letter_spacing"],
            "font_bold": [[False] * len(example["text"][i]) for i in range(length)],
            "font_italic": [[False] * len(example["text"][i]) for i in range(length)],
            "text_color": text_color,
            "text_line": text_line,
            "image": example["image"],
        }


def _decode_class_label(features: datasets.Features, example: dict) -> dict:
    """Apply `int2str` to all `datasets.ClassLabel` features in `example`."""

    def _get_decode_fn(feature: Any) -> Callable:
        if isinstance(feature, datasets.ClassLabel):
            return feature.int2str
        elif isinstance(feature, datasets.Sequence):
            return _get_decode_fn(feature.feature)
        return lambda x: x

    output = {}
    for key, feature in features.items():
        decode_fn = _get_decode_fn(feature)
        output[key] = decode_fn(example[key])
    return output


def _render_to_surface(
    font_manager: FontManager,
    example: Dict[str, Any],
    max_size: int,
    render_text: bool = True,
    format: str = "jpeg",
    canvas_color: skia.Color4f | None = skia.ColorWHITE
) -> bytes:
    """Render an example to a surface and return as JPEG bytes."""
    canvas_width = example["canvas_width"]
    canvas_height = example["canvas_height"]
    scale, size = _get_scale_size(canvas_width, canvas_height, max_size)
    surface = skia.Surface(size[0], size[1])
    with surface as canvas:
        canvas.scale(scale[0], scale[1])

        # if canvas_color is None, the background is transparent.
        if canvas_color is not None:
            canvas.clear(canvas_color)

        for i in range(example["length"]):
            with skia.AutoCanvasRestore(canvas):
                canvas.translate(example["left"][i], example["top"][i])
                if example["angle"][i] != 0.0:
                    canvas.rotate(
                        example["angle"][i],
                        example["width"][i] / 2.0,
                        example["height"][i] / 2.0,
                    )
                if example["type"][i] == "TextElement" and font_manager and render_text:
                    # TODO: Support other attributes like outline, shadow, etc.
                    element = TextElement(
                        uuid="",
                        type="textElement",
                        width=float(example["width"][i]),
                        height=float(example["height"][i]),
                        text=str(example["text"][i]),
                        fontSize=float(example["font_size"][i]),
                        font=str(example["font"][i]),
                        lineHeight=float(example["line_height"][i]),
                        textAlign=str(example["text_align"][i]),  # type: ignore
                        capitalize=bool(example["capitalize"][i]),
                        letterSpacing=float(example["letter_spacing"][i]),
                        boldMap=text_utils.generate_map(example["font_bold"][i]),
                        italicMap=text_utils.generate_map(example["font_italic"][i]),
                        colorMap=text_utils.generate_map(example["text_color"][i]),
                        lineMap=text_utils.generate_map(example["text_line"][i]),
                    )
                    text_utils.render_text(canvas, font_manager, element)
                else:
                    image = image_utils.convert_pil_image_to_skia_image(
                        example["image"][i]
                    )
                    src = skia.Rect(image.width(), image.height())
                    dst = skia.Rect(example["width"][i], example["height"][i])
                    paint = skia.Paint(Alphaf=example["opacity"][i], AntiAlias=True)
                    canvas.drawImageRect(image, src, dst, paint=paint)
    return image_utils.encode_surface(surface, format)


def _get_scale_size(
    width: float, height: float, max_size: Optional[int] = None
) -> Tuple[Tuple[float, float], Tuple[int, int]]:
    """Get scale factor for the image."""
    s = 1.0 if max_size is None else min(1.0, max_size / max(width, height))
    sx, sy = s, s
    # Ensure the size is not zero.
    if round(width * sx) <= 0:
        sx = 1.0 / width
    if round(height * sy) <= 0:
        sy = 1.0 / height
    size = (round(sx * width), round(sy * height))
    assert (
        size[0] > 0 and size[1] > 0
    ), f"Failed to get scale: {size=}, {width=}, {height=}"
    return (sx, sy), size
