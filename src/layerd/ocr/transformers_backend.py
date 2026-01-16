"""Transformers-based OCR backend using GOT-OCR2.

⚠️  CUDA REQUIRED: This backend requires NVIDIA GPU with CUDA support.
For CPU-compatible text detection, use the EAST backend instead.
"""

import importlib.util
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseOCR
from .types import BoundingBox, OCRBlock, OCRResult


class TransformersBackend(BaseOCR):
    """HuggingFace transformers-based OCR backend using GOT-OCR2.

    Full OCR with text recognition using the GOT-OCR2 model family.

    ⚠️  **CUDA REQUIRED**: GOT-OCR2's implementation requires CUDA (hardcoded .cuda() calls).
    For CPU inference, use the EAST backend instead: build_ocr('east', device='cpu')

    Features:
        - Full OCR with text recognition
        - Model size: ~1.4 GB
        - Device support: CUDA only (CPU not supported)
        - Requires: pip install layerd[ocr]

    Args:
        model_name: HuggingFace model identifier
            - "stepfun-ai/GOT-OCR2_0" (recommended, general OCR)
            - "ucaslcl/GOT-OCR-2.0-hf" (alternative GOT-OCR)
        device: Device to run on (must be "cuda" or "cuda:N")
        torch_dtype: Optional dtype override (default: float16 for CUDA)
        **kwargs: Additional model initialization parameters

    Raises:
        ImportError: If optional OCR dependencies are not installed
        ValueError: If device is not CUDA
        RuntimeError: If CUDA is not available

    Example:
        >>> from layerd.ocr import build_ocr
        >>> ocr = build_ocr("transformers", device="cuda")
        >>> result = ocr("design.png")
        >>> for block in result['blocks']:
        ...     print(f"Text: {block['text']}")

        >>> # For CPU, use EAST instead
        >>> ocr = build_ocr("east", device="cpu")
    """

    def __init__(
        self,
        model_name: str = "stepfun-ai/GOT-OCR2_0",
        device: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize transformers backend.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cuda" or "cuda:N")
            torch_dtype: Optional dtype override (default: float16 for CUDA)
            **kwargs: Additional model initialization parameters

        Raises:
            ImportError: If optional dependencies are not installed
            ValueError: If device is CPU (not supported)
            RuntimeError: If CUDA is not available
        """
        # Check for optional dependencies
        if importlib.util.find_spec("transformers") is None:
            raise ImportError(
                "Transformers backend requires optional dependencies. "
                "Install with: pip install layerd[ocr]"
            )

        # GOT-OCR2 requires CUDA (hardcoded in model.chat())
        if device == "cpu":
            raise ValueError(
                "GOT-OCR2's transformers backend requires CUDA. "
                "For CPU inference, use EAST backend: build_ocr('east', device='cpu')"
            )

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please install CUDA-enabled PyTorch or use EAST backend for CPU."
            )

        super().__init__(device=device)

        self.model_name = model_name

        # Auto-select dtype based on device
        if torch_dtype is None:
            torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.torch_dtype = torch_dtype

        # Load tokenizer and model (GOT-OCR2 official API)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            pad_token_id=None,  # Will be set from tokenizer
            **kwargs,
        ).to(device)

        # Set pad_token_id after loading
        if self.tokenizer.eos_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

    def infer(self, image: Image.Image, ocr_type: str = "ocr", **kwargs: Any) -> OCRResult:
        """Perform OCR inference using GOT-OCR2's official API.

        Args:
            image: PIL Image
            ocr_type: OCR type for GOT-OCR2 (default: "ocr")
                - "ocr": Plain text OCR
                - "format": Formatted text with layout
                - "box": With bounding boxes
            **kwargs: Additional parameters for model.chat()

        Returns:
            Standardized OCR result
        """
        # Save image temporarily (GOT-OCR2 expects file path)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Use GOT-OCR2's official chat API
            text_output = self.model.chat(self.tokenizer, tmp_path, ocr_type=ocr_type)
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        # Parse to standardized format
        result = self._parse_model_output(text_output, image.width, image.height)

        return result

    def _parse_model_output(self, text: str, image_width: int, image_height: int) -> OCRResult:
        """Parse model output text into structured OCRResult.

        This is a simple implementation that treats all text as one block.
        TODO: Add model-specific parsers for better structure extraction.

        Args:
            text: Raw model output text
            image_width: Original image width
            image_height: Original image height

        Returns:
            Standardized OCR result
        """
        # Split by newlines to get individual text blocks
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

        blocks: list[OCRBlock] = []

        if not lines:
            # Empty result
            return OCRResult(
                image_size=(image_width, image_height),
                blocks=[],
                metadata={"model": self.model_name, "backend": "transformers"},
            )

        # Simple heuristic: treat each non-empty line as a separate block
        # TODO: Extract bounding boxes from model-specific outputs
        block_height = image_height // max(len(lines), 1)

        for i, line in enumerate(lines):
            # Create simple top-to-bottom blocks spanning full width
            bbox = BoundingBox(
                x_min=0, y_min=i * block_height, x_max=image_width, y_max=min((i + 1) * block_height, image_height)
            )

            block = OCRBlock(text=line, bbox=bbox)

            blocks.append(block)

        return OCRResult(
            image_size=(image_width, image_height),
            blocks=blocks,
            metadata={"model": self.model_name, "backend": "transformers"},
        )

    def to(self, device: str) -> "TransformersBackend":
        """Move model to specified device.

        Args:
            device: Device string ('cuda', 'cuda:0', 'cuda:1', etc.)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If device is CPU (not supported)
        """
        if device == "cpu":
            raise ValueError("TransformersBackend does not support CPU. Use EAST backend for CPU inference.")

        self.device = device
        self.model = self.model.to(device)
        return self
