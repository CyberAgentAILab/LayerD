"""Abstract base class for OCR backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import fsspec
import numpy as np
from PIL import Image

from .types import OCRResult


class BaseOCR(ABC):
    """Abstract base class for OCR backends.

    Follows LayerD's pattern with template method (__call__ wraps infer())
    similar to BaseMatting and BaseInpaint.

    Supports input from:
    - PIL Images (RGB or RGBA)
    - Numpy arrays (H, W, C format)
    - File paths (local, GCS gs://, S3 s3://, HTTP, etc. via fsspec)

    Example:
        >>> from layerd.ocr import build_ocr
        >>> ocr = build_ocr(backend="east", device="cpu")
        >>>
        >>> # From PIL Image
        >>> result = ocr(image)
        >>>
        >>> # From file path
        >>> result = ocr("path/to/image.png")
        >>>
        >>> # From GCS
        >>> result = ocr("gs://bucket/image.png")
        >>>
        >>> # Batch processing
        >>> results = ocr([img1, img2, img3])
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize OCR backend.

        Args:
            device: Device to run on ("cuda", "cpu", "mps")
        """
        self.device = device

    def __call__(
        self,
        images: Union[Image.Image, list[Image.Image], np.ndarray, list[np.ndarray], str, Path, list[Union[str, Path]]],
        **kwargs: Any,
    ) -> Union[OCRResult, list[OCRResult]]:
        """Run OCR on image(s) with input validation.

        Template method pattern: validates inputs, delegates to infer(), validates outputs.

        Args:
            images: Single image or batch of images (PIL Image, numpy array, or file path)
            **kwargs: Backend-specific options

        Returns:
            OCRResult for single image, or list of OCRResults for batch
        """
        # Normalize inputs
        single_image = not isinstance(images, list)
        if single_image:
            images = [images]  # type: ignore

        # Convert to PIL Images
        pil_images: list[Image.Image] = []
        for img in images:  # type: ignore
            pil_img = self._normalize_input(img)
            pil_images.append(pil_img)

        # Delegate to subclass
        if single_image:
            result = self.infer(pil_images[0], **kwargs)
            self._validate_output(result)
            return result
        else:
            results = self.infer_batch(pil_images, **kwargs)
            for result in results:
                self._validate_output(result)
            return results

    @abstractmethod
    def infer(self, image: Image.Image, **kwargs: Any) -> OCRResult:
        """Perform OCR inference on a single image.

        Subclasses must implement this method.

        Args:
            image: PIL Image in RGB mode
            **kwargs: Backend-specific parameters

        Returns:
            Standardized OCR result
        """
        pass

    def infer_batch(self, images: list[Image.Image], **kwargs: Any) -> list[OCRResult]:
        """Perform batch OCR inference.

        Default implementation: sequential processing.
        Backends can override for true parallel batch processing.

        Args:
            images: List of PIL Images
            **kwargs: Backend-specific parameters

        Returns:
            List of standardized OCR results
        """
        return [self.infer(img, **kwargs) for img in images]

    def to(self, device: str) -> "BaseOCR":
        """Move model to specified device (for PyTorch-based backends).

        Enables method chaining: ocr.to("cuda:1").infer(image)

        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            Self for method chaining
        """
        self.device = device
        return self

    def _normalize_input(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        """Convert various input formats to PIL Image.

        Supports local files, GCS (gs://), S3 (s3://), HTTP, and other fsspec filesystems.

        Args:
            image: Input in various formats

        Returns:
            PIL Image in RGB mode

        Raises:
            TypeError: If input type is not supported
            FileNotFoundError: If image path does not exist
        """
        if isinstance(image, (str, Path)):
            # Load from file path (supports local, GCS, S3, HTTP via fsspec)
            path_str = str(image)
            with fsspec.open(path_str, "rb") as f:
                pil_img = Image.open(f)
                # Force image data to be read while the fsspec file handle is still open.
                # PIL loads lazily by default, so without this, later access could fail after
                # the context manager closes the underlying file or delay I/O/decoding errors.
                pil_img.load()
                return pil_img.convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL
            if image.ndim != 3 or image.shape[2] not in [3, 4]:
                raise ValueError(f"Expected (H,W,3) or (H,W,4) array, got shape {image.shape}")
            pil_img = Image.fromarray(image)
            return pil_img.convert("RGB")
        elif isinstance(image, Image.Image):
            # Already PIL Image, ensure RGB
            return image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image, numpy array, or file path.")

    def _validate_output(self, result: OCRResult) -> None:
        """Validate OCR result structure.

        Args:
            result: OCR result to validate

        Raises:
            TypeError: If result structure is invalid
            ValueError: If required fields are missing
        """
        if not isinstance(result, dict):
            raise TypeError(f"Expected OCRResult dict, got {type(result)}")

        # Check required fields
        if "image_size" not in result:
            raise ValueError("OCR result missing required field 'image_size'")
        if "blocks" not in result:
            raise ValueError("OCR result missing required field 'blocks'")

        # Validate image_size
        if not isinstance(result["image_size"], tuple) or len(result["image_size"]) != 2:
            raise ValueError(f"image_size must be tuple of (width, height), got {result['image_size']}")

        # Validate blocks
        if not isinstance(result["blocks"], list):
            raise TypeError(f"blocks must be a list, got {type(result['blocks'])}")

        for i, block in enumerate(result["blocks"]):
            if not isinstance(block, dict):
                raise TypeError(f"blocks[{i}] must be dict, got {type(block)}")
            if "text" not in block:
                raise ValueError(f"blocks[{i}] missing required field 'text'")
            if "bbox" not in block:
                raise ValueError(f"blocks[{i}] missing required field 'bbox'")
