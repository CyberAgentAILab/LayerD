"""PSD (Photoshop Document) export for LayerD elements."""

from io import BytesIO
from typing import Literal

from psd_tools import PSDImage

from layerd.types import Element

from .base import BaseExporter


class PSDBuilder(BaseExporter[bytes]):
    """PSD (Photoshop Document) export builder for LayerD elements.

    Exports decomposed layers as Photoshop-compatible PSD file format.
    Each element becomes a separate layer with metadata preserved.

    Args:
        color_depth: Bit depth per channel (8, 16, or 32)
        compression: Compression type for layer data ("rle", "zip", or "raw")

    Example:
        >>> builder = PSDBuilder()
        >>> psd_bytes = builder(elements, canvas_size)
        >>> with open("output.psd", "wb") as f:
        ...     f.write(psd_bytes)
    """

    def __init__(self, color_depth: Literal[8, 16, 32] = 8, compression: str = "rle") -> None:
        """Initialize PSD builder.

        Args:
            color_depth: Bit depth (8, 16, or 32)
            compression: Compression method ("rle", "zip", or "raw")
        """
        super().__init__()
        self.color_depth = color_depth
        self.compression = compression

    def export(self, elements: list[Element], canvas_size: tuple[int, int]) -> bytes:
        """Export elements to PSD bytes.

        Args:
            elements: List of processed elements with images and bounding boxes
            canvas_size: Canvas dimensions (width, height)

        Returns:
            PSD file as bytes
        """
        # Map compression string to psd-tools enum value
        compression_map = {
            "raw": 0,  # No compression
            "rle": 1,  # RLE compression (default, widely supported)
            "zip": 2,  # ZIP compression
        }
        compression_value = compression_map.get(self.compression.lower(), 1)

        # Create new PSD document
        psd = PSDImage.new(mode="RGBA", size=canvas_size, depth=self.color_depth)

        # Add layers in natural order
        # Elements are top-to-bottom (element[0] = topmost in visual stacking)
        # Photoshop renders layers in the same order for correct visual result

        # Calculate padding width based on max ID to handle >99 layers
        if elements:
            max_id = max(e["id"] for e in elements)
            pad_width = max(2, len(str(max_id)))
        else:
            pad_width = 2

        for element in elements:
            layer_name = f"{element['type']}_{element['id']:0{pad_width}d}"
            left = element["box"]["x_min"]
            top = element["box"]["y_min"]

            # Create pixel layer from element image
            psd.create_pixel_layer(
                image=element["image"],  # PIL.Image RGBA
                name=layer_name,
                left=left,
                top=top,
            )

        # Save to bytes with compression
        buffer = BytesIO()
        psd.save(buffer, compression=compression_value)
        return buffer.getvalue()
