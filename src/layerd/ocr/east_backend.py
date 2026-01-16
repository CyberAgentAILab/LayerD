"""EAST text detector backend for OCR (lightweight, CPU-compatible)."""

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from .base import BaseOCR
from .types import BoundingBox, OCRBlock, OCRResult, Polygon

logger = logging.getLogger(__name__)

# EAST model configuration
DEFAULT_MODEL_URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
EXPECTED_MODEL_SIZE = 97 * 1024 * 1024  # ~97 MB


def _get_default_model_path() -> Path:
    """Get default model path from environment or use default location.

    Environment variable: LAYERD_EAST_MODEL_PATH

    Returns:
        Path to EAST model file
    """
    env_path = os.environ.get("LAYERD_EAST_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return Path.home() / ".cache" / "layerd" / "east_detector.pb"


class EASTBackend(BaseOCR):
    """OpenCV EAST text detector backend.

    Lightweight text detection using OpenCV's EAST (Efficient and Accurate Scene Text) detector.
    Provides bounding box detection without text recognition (sufficient for LayerD pipeline).

    Features:
        - Model size: 97 MB (vs. 1.4 GB for GOT-OCR2)
        - Device support: CPU and CUDA
        - Platform support: Windows, Linux, macOS
        - Output: Text bounding boxes (no recognition)

    Args:
        model_path: Path to EAST model file. If None, uses default location
            (configurable via LAYERD_EAST_MODEL_PATH environment variable,
            defaults to ~/.cache/layerd/east_detector.pb)
        device: Device to run on ("cpu" or "cuda")
        conf_threshold: Confidence threshold for detection (0.0-1.0)
        nms_threshold: Non-maximum suppression threshold (0.0-1.0)
        input_width: Input image width (must be multiple of 32)
        input_height: Input image height (must be multiple of 32)

    Environment Variables:
        LAYERD_EAST_MODEL_PATH: Custom path for EAST model file

    Example:
        >>> from layerd.ocr import build_ocr
        >>> ocr = build_ocr("east", device="cpu")
        >>> result = ocr("design.png")
        >>> print(f"Detected {len(result['blocks'])} text regions")

        >>> # Custom model path via environment variable
        >>> import os
        >>> os.environ["LAYERD_EAST_MODEL_PATH"] = "/custom/path/east_detector.pb"
        >>> ocr = build_ocr("east", device="cpu")
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_width: int = 320,
        input_height: int = 320,
    ) -> None:
        super().__init__(device=device)

        # Validate input dimensions (must be multiples of 32)
        if input_width % 32 != 0 or input_height % 32 != 0:
            raise ValueError(f"Input dimensions must be multiples of 32, got {input_width}x{input_height}")

        # Download model if needed
        if model_path is None:
            model_path_obj = self._ensure_model_downloaded()
        else:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading EAST model from {model_path_obj}")

        # Load EAST model using high-level API
        try:
            self.detector = cv2.dnn.TextDetectionModel_EAST(str(model_path_obj))  # type: ignore[attr-defined]
        except cv2.error as e:
            raise RuntimeError(
                f"Failed to load EAST model from {model_path_obj}. "
                f"Error: {e}. "
                f"This may indicate a corrupted model file. Try deleting {model_path_obj} and re-running."
            ) from e

        # Set detection thresholds
        self.detector.setConfidenceThreshold(conf_threshold)
        self.detector.setNMSThreshold(nms_threshold)

        # Set input preprocessing parameters
        # EAST expects BGR input with specific mean subtraction
        self.detector.setInputParams(
            scale=1.0,
            size=(input_width, input_height),
            mean=(123.68, 116.78, 103.94),  # ImageNet mean (BGR order)
            swapRB=True,  # Convert RGB to BGR
        )

        # Configure device backend
        if device == "cpu":
            self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("EAST detector configured for CPU")
        elif device.startswith("cuda"):
            try:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("EAST detector configured for CUDA")
            except cv2.error as e:
                logger.warning(f"Failed to configure CUDA backend: {e}. Falling back to CPU.")
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            raise ValueError(f"Unsupported device: {device}. Use 'cpu' or 'cuda'")

        logger.info(
            f"EAST detector initialized (conf={conf_threshold}, nms={nms_threshold}, "
            f"input_size={input_width}x{input_height})"
        )

    def infer(self, image: Image.Image, **kwargs: Any) -> OCRResult:
        """Perform text detection on a single image.

        Args:
            image: PIL Image in RGB mode
            **kwargs: Unused (for interface compatibility)

        Returns:
            OCR result with detected text bounding boxes (text field will be empty)
        """
        # Convert PIL Image to OpenCV format (RGB -> BGR)
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Run EAST detection
        # Returns: list of rotated rectangles and confidence scores
        try:
            detections, confidences = self.detector.detect(img_cv)
        except cv2.error as e:
            logger.warning(f"EAST detection failed: {e}. Returning empty result.")
            return OCRResult(
                image_size=(image.width, image.height),
                blocks=[],
                metadata={"backend": "east", "model": "EAST", "error": str(e)},
            )

        # Convert detections to standardized format
        blocks: list[OCRBlock] = []

        if detections is not None and len(detections) > 0:
            for detection, confidence in zip(detections, confidences):
                # detection is a rotated rectangle: 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Convert to axis-aligned bounding box
                x_coords = [pt[0] for pt in detection]
                y_coords = [pt[1] for pt in detection]

                x_min = max(0, int(min(x_coords)))
                y_min = max(0, int(min(y_coords)))
                x_max = min(image.width, int(max(x_coords)))
                y_max = min(image.height, int(max(y_coords)))

                # Skip invalid boxes
                if x_max <= x_min or y_max <= y_min:
                    continue

                bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

                # Also store polygon for rotated text (LayerOrganizer supports this)
                polygon = [Polygon(x=int(pt[0]), y=int(pt[1])) for pt in detection]

                block = OCRBlock(
                    text="",  # EAST doesn't recognize text, only detects regions
                    bbox=bbox,
                    confidence=float(confidence),
                    polygon=polygon,
                )
                blocks.append(block)

        logger.info(f"EAST detected {len(blocks)} text regions")

        return OCRResult(
            image_size=(image.width, image.height),
            blocks=blocks,
            metadata={"backend": "east", "model": "EAST", "detections": len(blocks)},
        )

    def _ensure_model_downloaded(self) -> Path:
        """Download EAST model if not already cached.

        Returns:
            Path to the downloaded model file

        Raises:
            RuntimeError: If download fails
        """
        model_path = _get_default_model_path()

        if model_path.exists():
            # Verify file size
            file_size = model_path.stat().st_size
            if file_size < EXPECTED_MODEL_SIZE * 0.9:  # Allow 10% variance
                logger.warning(
                    f"Model file size ({file_size / 1024 / 1024:.1f} MB) is smaller than expected "
                    f"({EXPECTED_MODEL_SIZE / 1024 / 1024:.1f} MB). Re-downloading."
                )
                model_path.unlink()
            else:
                logger.info(f"Using cached EAST model from {model_path}")
                return model_path

        # Download model
        logger.info(f"Downloading EAST model from {DEFAULT_MODEL_URL}")
        logger.info(f"This is a one-time download (~97 MB) and will be cached at {model_path}")

        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(DEFAULT_MODEL_URL, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with open(model_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading EAST model") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"EAST model downloaded successfully to {model_path}")
            return model_path

        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()

            raise RuntimeError(
                f"Failed to download EAST model from {DEFAULT_MODEL_URL}. "
                f"Error: {e}. "
                f"You can manually download the model and specify the path using model_path parameter."
            ) from e
