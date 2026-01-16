# LayerD OCR Module

Optical Character Recognition (OCR) support for the LayerD pipeline with two backend options.

## Overview

The OCR module provides text detection and recognition capabilities for design decomposition. Two backends are available:

| Backend | Model Size | Device Support | Text Recognition | Installation |
|---------|-----------|----------------|------------------|--------------|
| **EAST** | ~97 MB | CPU, CUDA | ❌ (detection only) | No extras needed |
| **Transformers** | ~1.4 GB | CUDA only | ✅ (full OCR) | `pip install layerd[ocr]` |

## Installation

### EAST Backend (Default)

No additional dependencies required. EAST uses OpenCV which is already included in LayerD core:

```bash
pip install layerd
```

### Transformers Backend (GOT-OCR2)

Requires optional dependencies:

```bash
pip install layerd[ocr]
```

**Requirements:**
- NVIDIA GPU with CUDA support
- transformers ≤ 4.48.0 (version constraint for GOT-OCR2 compatibility)
- verovio ≥ 4.3.1

⚠️ **Note:** CPU inference is not supported by the transformers backend due to limitations in GOT-OCR2's implementation. Use EAST for CPU-compatible text detection.

## Quick Start

### Basic Usage (EAST)

```python
from layerd.ocr import build_ocr
from PIL import Image

# Initialize EAST detector (CPU-compatible)
ocr = build_ocr("east", device="cpu")

# Run OCR on image
image = Image.open("design.png")
result = ocr(image)

# Access results
print(f"Detected {len(result['blocks'])} text regions")
for block in result['blocks']:
    print(f"Box: {block['bbox']}")  # BoundingBox coordinates
    print(f"Confidence: {block.get('confidence', 'N/A')}")
```

### Full OCR with Text Recognition (Transformers)

```python
from layerd.ocr import build_ocr
from PIL import Image

# Initialize GOT-OCR2 (CUDA required)
ocr = build_ocr("transformers", device="cuda")

# Run OCR
image = Image.open("design.png")
result = ocr(image)

# Access recognized text
for block in result['blocks']:
    print(f"Text: {block['text']}")
    print(f"Box: {block['bbox']}")
```

### Integration with LayerDPipeline

```python
from layerd import LayerDPipeline
from PIL import Image

# Pipeline with EAST OCR (CPU-compatible)
pipeline = LayerDPipeline(ocr_backend="east", device="cpu")
result = pipeline(Image.open("design.png"))

# Access OCR results
if result.ocr_result:
    print(f"Found {len(result.ocr_result['blocks'])} text blocks")

# Export (text elements will be marked with type="text")
result.save("output.svg")
```

## Backend Comparison

### EAST Backend

**Pros:**
- Lightweight model (~97 MB)
- CPU and CUDA support
- Fast inference
- Works on all platforms (Windows, Linux, macOS)

**Cons:**
- No text recognition (bounding boxes only)
- Lower accuracy on rotated/curved text

**Best for:**
- CPU-only environments
- Quick text detection for layout analysis
- LayerD pipeline (detection sufficient for element organization)

### Transformers Backend (GOT-OCR2)

**Pros:**
- Full OCR with text recognition
- High accuracy
- Supports multiple OCR types ("ocr", "format", "box")

**Cons:**
- Large model (~1.4 GB)
- CUDA required (no CPU support)
- Slower inference
- Version constraint (transformers ≤ 4.48.0)

**Best for:**
- CUDA-enabled environments
- When text content is needed
- High-accuracy OCR requirements

## API Reference

### Factory Function

```python
def build_ocr(
    backend: Literal["east", "transformers"] = "east",
    device: str = "cpu",
    **kwargs
) -> BaseOCR
```

**Parameters:**
- `backend`: OCR backend ("east" or "transformers")
- `device`: Device for inference ("cpu", "cuda", "cuda:0", etc.)
- `**kwargs`: Backend-specific parameters

**Returns:** Initialized OCR backend

**Raises:**
- `ValueError`: If backend not supported or invalid device for backend
- `ImportError`: If optional dependencies not installed (transformers backend)

### OCR Result Format

```python
class OCRResult(TypedDict):
    image_size: tuple[int, int]       # (width, height)
    blocks: list[OCRBlock]             # Detected text blocks
    metadata: NotRequired[dict]        # Optional backend metadata

class OCRBlock(TypedDict):
    text: str                          # Recognized text (empty for EAST)
    bbox: BoundingBox                  # Rectangular bounding box
    confidence: NotRequired[float]     # Detection confidence (0.0-1.0)
    polygon: NotRequired[list[Polygon]]  # Rotated text polygon
    block_type: NotRequired[str]       # Classification (future use)

class BoundingBox(TypedDict):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
```

### BaseOCR Interface

```python
class BaseOCR:
    def __call__(self, images, **kwargs) -> OCRResult | list[OCRResult]:
        """Run OCR on single image or batch."""

    def infer(self, image: Image.Image, **kwargs) -> OCRResult:
        """Process single image (implement in subclasses)."""

    def infer_batch(self, images: list[Image.Image], **kwargs) -> list[OCRResult]:
        """Process batch of images."""

    def to(self, device: str) -> "BaseOCR":
        """Move model to specified device."""
```

## Advanced Usage

### Batch Processing

```python
ocr = build_ocr("east", device="cpu")
images = [Image.open(f"image_{i}.png") for i in range(10)]
results = ocr(images)  # Returns list[OCRResult]
```

### Custom EAST Model Path

```python
import os

# Via environment variable
os.environ["LAYERD_EAST_MODEL_PATH"] = "/custom/path/east_detector.pb"
ocr = build_ocr("east", device="cpu")

# Or via parameter
ocr = build_ocr("east", device="cpu", model_path="/custom/path/east_detector.pb")
```

### Device Switching

```python
ocr = build_ocr("east", device="cpu")
ocr = ocr.to("cuda")  # Move to GPU
```

### Custom Detection Parameters (EAST)

```python
ocr = build_ocr(
    "east",
    device="cpu",
    conf_threshold=0.7,      # Higher threshold = fewer false positives
    nms_threshold=0.3,       # Lower threshold = more aggressive suppression
    input_width=640,         # Larger = better for small text (must be multiple of 32)
    input_height=640,
)
```

### GOT-OCR2 OCR Types

```python
ocr = build_ocr("transformers", device="cuda")

# Plain text OCR (default)
result = ocr(image, ocr_type="ocr")

# Formatted text with layout
result = ocr(image, ocr_type="format")

# With bounding boxes
result = ocr(image, ocr_type="box")
```

## File System Support

The OCR module supports multiple file systems via fsspec:

```python
# Local files
result = ocr("path/to/image.png")

# Google Cloud Storage
result = ocr("gs://bucket/image.png")

# Amazon S3
result = ocr("s3://bucket/image.png")

# HTTP
result = ocr("https://example.com/image.png")

# PIL Images
image = Image.open("design.png")
result = ocr(image)

# Numpy arrays
import numpy as np
img_array = np.array(image)
result = ocr(img_array)
```

## Troubleshooting

### "CUDA is not available" Error

**Problem:** Transformers backend requires CUDA but it's not available.

**Solution:**
- Use EAST backend for CPU: `build_ocr("east", device="cpu")`
- Or install CUDA-enabled PyTorch

### "transformers OCR backend requires: pip install layerd[ocr]"

**Problem:** Optional dependencies not installed.

**Solution:**
```bash
pip install layerd[ocr]
```

### "transformers backend requires CUDA"

**Problem:** Trying to use transformers backend on CPU.

**Solution:**
- GOT-OCR2 has hardcoded CUDA calls and cannot run on CPU
- Use EAST backend instead: `build_ocr("east", device="cpu")`

### Model Download Issues

**Problem:** EAST model download fails or is slow.

**Solution:**
- The model (~97 MB) is automatically downloaded on first use
- It's cached at `~/.cache/layerd/east_detector.pb`
- You can manually download from: https://github.com/oyyd/frozen_east_text_detection.pb
- Then specify path: `build_ocr("east", model_path="/path/to/east_detector.pb")`

### Version Constraint Issues

**Problem:** transformers version conflicts.

**Solution:**
- GOT-OCR2 requires transformers ≤ 4.48.0
- If you need a newer transformers version, use EAST backend
- Or wait for GOT-OCR2 to support newer versions

## Performance Tips

1. **Use EAST for LayerD Pipeline**: Text recognition is not needed for layer organization
2. **Batch Processing**: Process multiple images together for better throughput
3. **Adjust EAST Input Size**: Larger sizes (640x640) improve accuracy but are slower
4. **Cache Models**: Models are automatically cached, but ensure cache directory is writable
5. **CUDA for Transformers**: Always use CUDA for transformers backend (CPU not supported)

## Examples

See the LayerD repository for complete examples:
- `examples/ocr_basic.py` - Basic OCR usage
- `examples/pipeline_with_ocr.py` - Integrated pipeline
- `tools/gradio_demo.py` - Interactive demo

## License

This module is part of LayerD and follows the same license. The EAST model is from https://github.com/oyyd/frozen_east_text_detection.pb.
