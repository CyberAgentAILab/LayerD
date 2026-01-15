"""SVG export and import for LayerD elements."""

import base64
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image

from layerd.types import BoundingBox, Element

from .base import BaseExporter


class SVGBuilder(BaseExporter[str]):
    """SVG export builder for LayerD elements.

    Supports two image embedding modes:
    - base64: Embed images as data URIs (default, self-contained)
    - external: Save images to directory and reference via relative paths

    The builder produces standards-compliant SVG with element metadata
    stored as data attributes (data-type, data-id) for round-trip conversion.

    Args:
        image_mode: Image embedding mode ("base64" or "external")
        image_dir: Directory for external images (required if image_mode="external")

    Example:
        >>> # Base64 mode (default)
        >>> builder = SVGBuilder()
        >>> svg = builder(elements, canvas_size)
        >>>
        >>> # External images
        >>> builder = SVGBuilder(image_mode="external", image_dir="./images")
        >>> svg = builder(elements, canvas_size)
    """

    def __init__(self, image_mode: Literal["base64", "external"] = "base64", image_dir: str | None = None) -> None:
        """Initialize SVG builder.

        Args:
            image_mode: "base64" for embedded images, "external" for file references
            image_dir: Directory path for external images (required if image_mode="external")

        Raises:
            ValueError: If image_mode="external" but image_dir is None
        """
        super().__init__()
        self.image_mode = image_mode
        self.image_dir = Path(image_dir) if image_dir else None

        if image_mode == "external" and self.image_dir is None:
            raise ValueError("image_dir must be provided when image_mode='external'")

        # Create directory if needed
        if self.image_dir is not None:
            self.image_dir.mkdir(parents=True, exist_ok=True)

    def export(self, elements: list[Element], canvas_size: tuple[int, int]) -> str:
        """Export elements to SVG string.

        Args:
            elements: List of processed elements
            canvas_size: Canvas dimensions (width, height)

        Returns:
            SVG string with embedded or linked images
        """
        width, height = canvas_size

        svg_parts = [
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        ]

        for elem in elements:
            # Get image href based on mode
            if self.image_mode == "external":
                href = self._save_element_image(elem)
            else:
                href = self._element_to_base64(elem)

            # Get bounding box
            box = elem["box"]
            x = box["x_min"]
            y = box["y_min"]
            w = box["x_max"] - box["x_min"]
            h = box["y_max"] - box["y_min"]

            # Create image element with metadata
            svg_parts.append(
                f'  <image x="{x}" y="{y}" width="{w}" height="{h}" '
                f'href="{href}" '
                f'data-type="{elem["type"]}" '
                f'data-id="{elem["id"]}" />'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _element_to_base64(self, element: Element) -> str:
        """Convert element image to base64 data URI.

        Args:
            element: Element with PIL Image

        Returns:
            Data URI string
        """
        buffered = BytesIO()
        element["image"].save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def _save_element_image(self, element: Element) -> str:
        """Save element image to disk and return relative path.

        Args:
            element: Element to save

        Returns:
            Relative path for SVG href attribute
        """
        assert self.image_dir is not None  # Validated in __init__

        filename = f"{element['id']}_{element['type']}.png"
        filepath = self.image_dir / filename
        element["image"].save(filepath, format="PNG")
        return f"./{self.image_dir.name}/{filename}"


class SVGParser:
    """SVG parser for converting SVG back to elements.

    Separate from SVGBuilder to follow single responsibility principle.
    Import/export are asymmetric operations with different concerns.

    Example:
        >>> parser = SVGParser()
        >>> elements = parser.parse(svg_content, svg_path="output.svg")
    """

    def parse(self, svg_content: str, svg_path: str | None = None) -> list[Element]:
        """Parse SVG string into elements.

        Args:
            svg_content: SVG string with embedded or linked images
            svg_path: Path to SVG file (required for external images)

        Returns:
            List of reconstructed elements

        Raises:
            ValueError: If external images found but svg_path not provided
        """
        # Parse SVG
        root = ET.fromstring(svg_content)

        # Handle namespace
        ns = {"svg": "http://www.w3.org/2000/svg"}
        image_tags = root.findall(".//svg:image", ns)
        if not image_tags:
            image_tags = root.findall(".//image")

        elements: list[Element] = []

        for img_tag in image_tags:
            # Extract attributes
            x = int(img_tag.get("x", "0"))
            y = int(img_tag.get("y", "0"))
            width = int(img_tag.get("width", "0"))
            height = int(img_tag.get("height", "0"))
            elem_type = img_tag.get("data-type", "image")
            elem_id = int(img_tag.get("data-id", "0"))

            # Extract and load image
            href = img_tag.get("href") or img_tag.get("{http://www.w3.org/1999/xlink}href")
            if href is None:
                raise ValueError("Image href is required for <image> elements in SVG")

            if href.startswith("data:image/png;base64,"):
                image = self._load_base64_image(href)
            else:
                if svg_path is None:
                    raise ValueError("svg_path required for external image references")
                image = self._load_external_image(href, svg_path)

            # Create bounding box
            bbox = BoundingBox(
                x_min=x,
                y_min=y,
                x_max=x + width,
                y_max=y + height,
            )

            # Create element
            elements.append(Element(id=elem_id, type=elem_type, image=image, box=bbox))

        return elements

    def _load_base64_image(self, href: str) -> Image.Image:
        """Load image from base64 data URI.

        Args:
            href: Data URI with base64-encoded image

        Returns:
            PIL Image
        """
        base64_data = href.replace("data:image/png;base64,", "")
        img_bytes = base64.b64decode(base64_data)
        return Image.open(BytesIO(img_bytes))

    def _load_external_image(self, href: str, svg_path: str) -> Image.Image:
        """Load image from external file.

        Args:
            href: Relative path to image file
            svg_path: Path to SVG file (for resolving relative paths)

        Returns:
            PIL Image
        """
        # Resolve relative path
        if href.startswith("./"):
            href = href[2:]
        svg_path_obj = Path(svg_path)
        image_path = svg_path_obj.parent / href
        return Image.open(image_path)

    def __call__(self, svg_content: str, svg_path: str | None = None) -> list[Element]:
        """Convenience method for parsing.

        Args:
            svg_content: SVG string
            svg_path: Path to SVG file

        Returns:
            List of elements
        """
        return self.parse(svg_content, svg_path)
