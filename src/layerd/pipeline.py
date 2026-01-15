"""Unified pipeline for end-to-end layer decomposition and export.

This module provides LayerDPipeline, a high-level API that orchestrates:
1. Layer decomposition (LayerD)
2. Layer organization (LayerOrganizer)
3. Element classification (ElementLabeler)

Export functionality (SVG, PSD) is provided through PipelineResult methods.

Example usage:
    >>> from layerd import LayerDPipeline
    >>> from PIL import Image
    >>> pipeline = LayerDPipeline()
    >>> image = Image.open("design.png")
    >>> result = pipeline(image)
    >>> result.save("output.svg")  # Auto-detects format
    >>> # OR: svg_string = result.to_svg()

Advanced usage with custom labeler:
    >>> from layerd import GradientAwareLabeler
    >>> pipeline = LayerDPipeline(
    ...     labeler=GradientAwareLabeler(entropy_threshold=5.0),
    ...     device="cuda"
    ... )
    >>> result = pipeline(image, max_iterations=3)
    >>> result.save("output.psd")  # PSD export

The pipeline is designed for future REST API server integration with Pydantic models
that support JSON serialization.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict

from layerd.classification import ElementLabeler, EntropyLabeler
from layerd.export import SVGBuilder, build_exporter
from layerd.models.layerd import LayerD
from layerd.postprocess import LayerOrganizer
from layerd.types import Element

logger = logging.getLogger(__name__)

# Sentinel value for "not provided" parameter
_UNSET = object()


class PipelineResult(BaseModel):
    """Result from LayerDPipeline processing.

    Uses Pydantic for future API server compatibility.
    Serializable to JSON for REST API responses (with custom serializers for images).

    Attributes:
        elements: Organized elements with bounding boxes and type classification
        layers: List of RGBA PIL Images (background + foreground layers)
        ocr_result: Reserved for future OCR integration (Phase 3)
        canvas_size: Original image size as (width, height) tuple
    """

    elements: list[Element]
    layers: list[Any]  # PIL Images - requires custom serializer for API use
    ocr_result: dict[str, Any] | None
    canvas_size: tuple[int, int]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_svg(
        self,
        image_mode: Literal["base64", "external"] = "base64",
        image_dir: str | None = None,
    ) -> str:
        """Generate SVG string representation.

        Args:
            image_mode: "base64" (embedded) or "external" (separate files)
            image_dir: Directory for external images (required if image_mode="external")

        Returns:
            SVG string

        Raises:
            ValueError: If image_mode="external" but image_dir not provided

        Example:
            >>> result = pipeline(image)
            >>> svg = result.to_svg()  # Base64 embedded
            >>> svg = result.to_svg(image_mode="external", image_dir="./images")
        """
        if image_mode == "external" and image_dir is None:
            raise ValueError("image_dir required when image_mode='external'")

        builder = SVGBuilder(image_mode=image_mode, image_dir=image_dir)
        return builder(self.elements, self.canvas_size)

    def to_psd(
        self,
        compression: str = "rle",
        color_depth: Literal[8, 16, 32] = 8,
    ) -> bytes:
        """Generate PSD bytes representation.

        Args:
            compression: Compression method ("rle" or "zip")
            color_depth: Bit depth per channel (8, 16, or 32)

        Returns:
            PSD file as bytes

        Example:
            >>> result = pipeline(image)
            >>> psd_bytes = result.to_psd()
            >>> with open("output.psd", "wb") as f:
            ...     f.write(psd_bytes)
        """
        builder = build_exporter("psd", compression=compression, color_depth=color_depth)
        return builder(self.elements, self.canvas_size)

    def save(
        self,
        path: str,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Save result to file.

        Format auto-detected from file extension if not specified.

        Args:
            path: Output file path
            format: Export format ("svg" or "psd"), auto-detected if None
            **kwargs: Format-specific options (passed to to_svg() or to_psd())

        Raises:
            ValueError: If format cannot be determined or is unsupported

        Examples:
            >>> result.save("output.svg")  # Auto-detects SVG format
            >>> result.save("output.psd")  # Auto-detects PSD format
            >>> result.save("output", format="svg")  # Explicit format
            >>> result.save("output.svg", image_mode="external", image_dir="./img")
        """
        # Auto-detect format from extension
        if format is None:
            suffix = Path(path).suffix.lower()
            if suffix == ".svg":
                format = "svg"
            elif suffix == ".psd":
                format = "psd"
            else:
                raise ValueError(
                    f"Cannot determine format from extension '{suffix}'. "
                    "Specify format explicitly with format='svg' or format='psd'"
                )

        # Generate data
        data: str | bytes
        mode: str
        if format == "svg":
            data = self.to_svg(**kwargs)
            mode = "w"
        elif format == "psd":
            data = self.to_psd(**kwargs)
            mode = "wb"
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'svg' or 'psd'")

        # Save to local file
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode) as f:
            f.write(data)


class LayerDPipeline:
    """Unified pipeline for end-to-end layer decomposition and export.

    Orchestrates the complete workflow from image input to export:
    - Layer decomposition with BiRefNet-based matting
    - Layer organization with element extraction
    - Element classification (text vs vector vs image)
    - Export to SVG or PSD formats

    The pipeline is stateless and thread-safe. Each call processes one image independently.

    Args:
        matting_hf_card: HuggingFace model card for matting model
        matting_process_size: Processing size for matting (width, height). None = auto
        matting_weight_path: Path to custom matting weights. None = use HF card
        use_unblend: Enable unblending for foreground color estimation
        bg_refine: Enable background refinement
        fg_refine: Enable foreground refinement
        fg_refine_num_colors: Number of colors for foreground refinement
        bg_refine_num_colors: Number of colors for background refinement
        kernel_scale: Kernel scale for refinement
        overlap_threshold: Overlap threshold for element organization
        labeler: Element classifier (None = disable, default = EntropyLabeler)
        labeler_threshold: Entropy threshold for default EntropyLabeler
        device: Device for computation ("cpu" or "cuda")

    Note:
        Device can be changed after initialization using the .to() method.

    Example:
        >>> pipeline = LayerDPipeline(device="cuda")
        >>> result = pipeline(image, max_iterations=3)
        >>> result.save("output.svg")
    """

    def __init__(
        self,
        # LayerD parameters
        matting_hf_card: str = "cyberagent/layerd-birefnet",
        matting_process_size: tuple[int, int] | None = None,
        matting_weight_path: str | None = None,
        use_unblend: bool = True,
        bg_refine: bool = True,
        fg_refine: bool = True,
        fg_refine_num_colors: int = 2,
        bg_refine_num_colors: int = 10,
        kernel_scale: float = 0.015,
        # LayerOrganizer parameters
        overlap_threshold: float = 0.9,
        labeler: ElementLabeler | None = _UNSET,  # type: ignore[assignment]  # Sentinel for "not provided"
        labeler_threshold: float = 5.0,
        # Device
        device: str = "cpu",
    ) -> None:
        """Initialize the pipeline with configuration parameters."""
        # Initialize LayerD model
        self.layerd = LayerD(
            matting_hf_card=matting_hf_card,
            matting_process_size=matting_process_size,
            matting_weight_path=matting_weight_path,
            use_unblend=use_unblend,
            bg_refine=bg_refine,
            fg_refine=fg_refine,
            fg_refine_num_colors=fg_refine_num_colors,
            bg_refine_num_colors=bg_refine_num_colors,
            kernel_scale=kernel_scale,
            device=device,
        )

        # Store LayerOrganizer parameters
        self.overlap_threshold = overlap_threshold

        # Store labeler (defaults to EntropyLabeler with specified threshold if not provided)
        self.labeler: ElementLabeler | None
        if labeler is _UNSET:
            self.labeler = EntropyLabeler(threshold=labeler_threshold)
        else:
            self.labeler = labeler  # type: ignore[assignment]  # labeler can be _UNSET at runtime

        # Store device
        self.device = device

    def __call__(
        self,
        image: Image.Image,
        max_iterations: int = 3,
    ) -> PipelineResult:
        """Run the complete pipeline on an image.

        Args:
            image: Input PIL Image (RGB or RGBA)
            max_iterations: Maximum number of decomposition iterations

        Returns:
            PipelineResult with elements, layers, and canvas size

        Note:
            SVG/PSD generation is done via PipelineResult.to_svg()/to_psd()/save()

        Raises:
            ValueError: If image is invalid or decomposition fails
            RuntimeError: If a pipeline stage fails unexpectedly
        """
        # Step 1: Decompose with LayerD
        try:
            layers = self.layerd.decompose(image, max_iterations=max_iterations)
            canvas_size = image.size  # (width, height)
        except Exception as e:
            raise RuntimeError(f"LayerD decomposition failed: {e}") from e

        # Step 2: Organize layers
        try:
            organizer = LayerOrganizer(
                overlap_threshold=self.overlap_threshold,
                labeler=self.labeler,
            )
            # No OCR in Phase 2 - will be added in Phase 3
            elements = organizer.organize(layers, ocr_result=None)
            logger.info(f"Organized {len(elements)} elements from {len(layers)} layers")
        except Exception as e:
            raise RuntimeError(f"Layer organization failed: {e}") from e

        # Step 3: Return result (SVG/PSD generation moved to PipelineResult methods)
        return PipelineResult(
            elements=elements,
            layers=layers,
            ocr_result=None,  # Reserved for Phase 3 OCR integration
            canvas_size=canvas_size,
        )

    def to(self, device: str) -> "LayerDPipeline":
        """Move pipeline to specified device.

        Args:
            device: Target device ("cpu" or "cuda")

        Returns:
            Self for method chaining

        Example:
            >>> pipeline = LayerDPipeline(device="cpu")
            >>> pipeline.to("cuda")  # Move to GPU
        """
        self.layerd = self.layerd.to(device)
        self.device = device
        return self
