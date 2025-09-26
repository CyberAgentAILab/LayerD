from typing import Any

import cr_renderer.image_utils as image_utils
import cr_renderer.text_utils as text_utils
import skia  # type: ignore
from cr_renderer.fonts import FontManager
from cr_renderer.renderer import CrelloV5Renderer, _decode_class_label
from cr_renderer.schema import TextElement
from PIL import Image


class CrelloV5RendererLayers(CrelloV5Renderer):
    """Overriden CrelloV5Renderer to expose layer-wise rendering."""

    def get_layer_types(self, example: dict[str, Any]) -> tuple[str, ...]:
        example = _decode_class_label(self.features, example)
        return tuple(example["type"])

    def get_is_transparent(self, example: dict[str, Any]) -> tuple[bool, ...]:
        example = _decode_class_label(self.features, example)
        return tuple(is_transparent(img) for img in example["image"])

    def render(
        self,
        example: dict[str, Any],
        short_side_size: int = 360,
        render_text: bool = True,
        layer_indices: list[int] | None = None,
    ) -> bytes:
        example = _decode_class_label(self.features, example)
        return _render_to_surface(self.font_manager, example, short_side_size, render_text, layer_indices=layer_indices)

    def render_layers(
        self, example: dict[str, Any], short_side_size: int = 360, render_text: bool = True
    ) -> list[bytes]:
        example = _decode_class_label(self.features, example)
        return _render_to_surface_layers(self.font_manager, example, short_side_size, render_text)


def is_transparent(image: Image.Image) -> bool:
    """Check if a PIL image has any transparency."""
    if image.mode != "RGBA":
        return False
    alpha_channel = image.split()[-1]
    return not alpha_channel.getextrema()[1] == 255


def _render_element_to_surface(
    example: dict[str, Any], i: int, canvas: skia.Canvas, font_manager: FontManager, render_text: bool = True
) -> skia.Canvas:
    """Render a single element to the given canvas."""
    with skia.AutoCanvasRestore(canvas):
        canvas.translate(example["left"][i], example["top"][i])
        if example["angle"][i] != 0.0:
            canvas.rotate(example["angle"][i], example["width"][i] / 2.0, example["height"][i] / 2.0)
        if example["type"][i] == "TextElement" and font_manager and render_text:
            element = TextElement(
                uuid="",  # ID is not required.
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
            image = image_utils.convert_pil_image_to_skia_image(example["image"][i])
            src = skia.Rect(image.width(), image.height())
            dst = skia.Rect(example["width"][i], example["height"][i])
            paint = skia.Paint(Alphaf=example["opacity"][i], AntiAlias=True)
            canvas.drawImageRect(image, src, dst, paint=paint)
    return canvas


def _render_to_surface(
    font_manager: FontManager,
    example: dict[str, Any],
    short_side_size: int,
    render_text: bool = True,
    layer_indices: list[int] | None = None,
) -> bytes:
    """Render an example to a surface and return as PNG bytes."""
    canvas_width = example["canvas_width"]
    canvas_height = example["canvas_height"]
    scale = short_side_size / min(canvas_width, canvas_height)
    size = (int(canvas_width * scale), int(canvas_height * scale))
    scale = (scale, scale)
    surface = skia.Surface(size[0], size[1])
    layer_indices = layer_indices if layer_indices is not None else list(range(example["length"]))
    with surface as canvas:
        canvas.scale(scale[0], scale[1])
        canvas.clear(skia.ColorWHITE)
        for i in layer_indices:
            canvas = _render_element_to_surface(example, i, canvas, font_manager, render_text)
    return image_utils.encode_surface(surface, "png")


def _render_to_surface_layers(
    font_manager: FontManager,
    example: dict[str, Any],
    short_side_size: int,
    render_text: bool = True,
) -> list[bytes]:
    """Render example layers to individual surfaces and return as PNG bytes."""
    canvas_width = example["canvas_width"]
    canvas_height = example["canvas_height"]
    scale = short_side_size / min(canvas_width, canvas_height)
    size = (int(canvas_width * scale), int(canvas_height * scale))
    scale = (scale, scale)
    layers = []
    for i in range(example["length"]):
        surface = skia.Surface(size[0], size[1])
        with surface as canvas:
            canvas.scale(scale[0], scale[1])
            canvas.clear(skia.ColorTRANSPARENT)
            canvas = _render_element_to_surface(example, i, canvas, font_manager, render_text)
            layer = image_utils.encode_surface(surface, "png")
            layers.append(layer)
    return layers
