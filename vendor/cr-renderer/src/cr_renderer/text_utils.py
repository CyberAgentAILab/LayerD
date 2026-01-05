import itertools
import logging
from typing import Any, Iterator, List, Literal, NamedTuple, Optional, Type, TypeVar

import skia  # type: ignore

from cr_renderer.color_utils import parse_color
from cr_renderer.fonts import FontManager
from cr_renderer.schema import (
    CurvedTextEffect,
    DropShadowTextEffect,
    EchoTextEffect,
    GlitchTextEffect,
    OutlineTextEffect,
    ReflectionTextEffect,
    TextBoxTextEffect,
    TextEffect,
    TextElement,
    TextMapItem,
    TextProperty,
)

logger = logging.getLogger(__name__)


def render_text(
    canvas: skia.Canvas, font_manager: FontManager, element: TextElement
) -> None:
    """Render text."""
    check_supported_effects(element)

    for line_index, line in enumerate(make_text_spans(element)):
        y = line_index * element.fontSize * (element.lineHeight or 1.0)
        blobs = [make_text_blob(element, span, font_manager) for span in line]
        x = get_textalign_offset(element.textAlign, element.width, blobs)

        render_text_box_effect(canvas, element, x, y, blobs)

        outline = find_text_effect(element, OutlineTextEffect)
        if outline is None or outline.hasInnerText:
            render_text_fill(canvas, element, x, y, blobs)

        if outline is not None:
            render_text_outline(canvas, element, outline, x, y, blobs)


def check_supported_effects(element: TextElement) -> None:
    curved = find_text_effect(element, CurvedTextEffect)
    if curved is not None:
        raise NotImplementedError(f"Curved effect is not supported: {curved}")

    for effect_type in [
        DropShadowTextEffect,
        GlitchTextEffect,
        ReflectionTextEffect,
        EchoTextEffect,
    ]:
        effect = find_text_effect(element, effect_type)  # type: ignore
        if effect is not None:
            logger.warning(f"{effect_type.__name__} is ignored: {effect}")

    if element.opacity < 1.0:
        logger.info(f"Text element opacity may be inaccurate: {element.opacity=}")


class TextSpanStyle(NamedTuple):
    color: str
    bold: bool
    italic: bool
    link: Optional[str]
    opacity: float
    underline: Optional[bool]
    weight: int
    style: str


class TextSpan(NamedTuple):
    text: str
    style: TextSpanStyle


def make_default_map(
    length: int, value: Optional[Any] = None, type: Optional[str] = None
) -> List[TextMapItem]:
    return [TextMapItem(startIndex=0, endIndex=length - 1, type=type, value=value)]


def make_dense_map(
    m: Optional[List[TextMapItem]], text_length: int, default: Any = None
) -> List[Any]:
    """Make a dense map from a run-length array."""
    if m is None:
        m = make_default_map(text_length, default)

    x = [default] * text_length
    invalid_flag = False
    for item in m:
        for i in range(item.startIndex, item.endIndex + 1):
            if i < text_length:
                x[i] = item.value
            else:
                invalid_flag = True

    if invalid_flag:
        logger.debug(f"Invalid run length array {m=} for {text_length=}")
    return x


def generate_map(values: List[Any], type: Optional[str] = None) -> List[TextMapItem]:
    """Generate a map from a list of values."""
    offset = 0
    result = []
    for value, it in itertools.groupby(values):
        length = len(list(it))
        result.append(
            TextMapItem(
                startIndex=offset,
                endIndex=offset + length - 1,
                type=type,
                value=value,
            )
        )
        offset += length
    return result


def make_text_spans(element: TextProperty) -> Iterator[List[TextSpan]]:
    """Generate a list of spans for each line of text.

    Example:

        for line in make_text_spans(text_element):
            for span in line:
                print(span.text, span.style)
    """

    def _make_line_spans(
        text: str, style_map: List[TextSpanStyle]
    ) -> Iterator[TextSpan]:
        offset = 0
        for style, it in itertools.groupby(style_map):
            length = len(list(it))
            yield TextSpan(text[offset : offset + length], style)
            offset += length

    text_length = len(element.text)
    style_map = list(
        map(
            TextSpanStyle,
            make_dense_map(element.colorMap, text_length),
            make_dense_map(element.boldMap, text_length, False),
            make_dense_map(element.italicMap, text_length, False),
            make_dense_map(element.linkMap, text_length, None),
            make_dense_map(element.opacityMap, text_length, 1.0),
            make_dense_map(element.underlineMap, text_length, None),
            make_dense_map(element.weightMap, text_length, 400),
            make_dense_map(element.styleMap, text_length, "regular"),
        )
    )

    for line in element.lineMap or make_default_map(text_length):
        line_text = element.text[line.startIndex : line.endIndex + 1]
        line_style = style_map[line.startIndex : line.endIndex + 1]
        yield list(_make_line_spans(line_text, line_style))


class BlobItem(NamedTuple):
    blob: skia.TextBlob
    width: float
    color: int
    metrics: skia.FontMetrics


def make_text_blob(
    element: TextProperty,
    span: TextSpan,
    font_manager: FontManager,
) -> BlobItem:
    """Render text blob on canvas, and returns horizontal advance."""
    text = span.text.upper() if element.capitalize else span.text
    text = text.strip("\n")

    font_weight = "regular"
    font_style = "regular"
    if span.style.bold and span.style.italic:
        font_weight = "bold"
        font_style = "bolditalic"
    elif span.style.bold:
        font_weight = "bold"
        font_style = "bold"
    elif span.style.italic:
        font_style = "italic"

    ttf_data = font_manager.lookup(element.font, font_weight, font_style)
    typeface = skia.Typeface.MakeFromData(ttf_data)
    font = skia.Font(typeface, int(element.fontSize))
    glyphs = font.textToGlyphs(text)

    positions = [
        xpos + (element.letterSpacing * index)
        for index, xpos in enumerate(font.getXPos(glyphs))
    ]
    builder = skia.TextBlobBuilder()
    builder.allocRunPosH(font, glyphs, positions, 0)
    blob = builder.make()
    color = parse_color(span.style.color, span.style.opacity)

    # TODO: Underline decoration.

    total_width = sum(font.getWidths(glyphs))
    total_width += element.letterSpacing * max(0, len(glyphs) - 1)
    return BlobItem(blob, total_width, color, font.getMetrics())


def get_textalign_offset(
    align: Literal["left", "right", "center", "justify"],
    width: float,
    blobs: List[BlobItem],
) -> float:
    blob_width = sum(blob.width for blob in blobs)
    if align == "left":
        return 0.0
    elif align == "center":
        return (width - blob_width) / 2.0
    elif align == "right":
        return width - blob_width
    else:
        logger.warning(f"Unsupported text align: {align}")
    return 0.0


T = TypeVar("T", bound=TextEffect)


def find_text_effect(element: TextElement, effect_type: Type[T]) -> Optional[T]:
    for effect in element.effects or []:
        if isinstance(effect, effect_type) and effect.enabled:  # type: ignore
            return effect
    return None


def render_text_box_effect(
    canvas: skia.Canvas,
    element: TextElement,
    x: float,
    y: float,
    blobs: List[BlobItem],
) -> None:
    effect = find_text_effect(element, TextBoxTextEffect)
    if effect is None:
        return

    width = sum(blob.width for blob in blobs)
    logger.debug(f"{effect.spread=} is ignored.")
    rrect = skia.RRect.MakeRectXY(
        skia.Rect.MakeXYWH(x, y, width, element.fontSize),
        effect.radius,
        effect.radius,
    )
    paint = skia.Paint(
        AntiAlias=True,
        Color=parse_color(effect.color, effect.blendOpacity),
    )
    canvas.drawRRect(rrect, paint)


def render_text_fill(
    canvas: skia.Canvas,
    element: TextElement,
    x: float,
    y: float,
    blobs: List[BlobItem],
) -> None:
    paint = skia.Paint(AntiAlias=True, Alphaf=float(element.opacity))
    for blob in blobs:
        paint.setColor(blob.color)
        canvas.drawTextBlob(
            blob=blob.blob,
            x=x,
            y=y - blob.metrics.fAscent,
            paint=paint,
        )
        x += blob.width

    if x > element.width:
        logger.info(f"blob_width={x} exceeds {element.width=}")


def render_text_outline(
    canvas: skia.Canvas,
    element: TextElement,
    effect: OutlineTextEffect,
    x: float,
    y: float,
    blobs: List[BlobItem],
) -> None:
    paint = skia.Paint(
        AntiAlias=True,
        Alphaf=float(element.opacity),
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=effect.thickness,
        Color=parse_color(effect.color, effect.opacity),
    )
    for blob in blobs:
        canvas.drawTextBlob(
            blob=blob.blob,
            x=x,
            y=y - blob.metrics.fAscent,
            paint=paint,
        )
        x += blob.width
