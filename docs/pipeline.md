# LayerDPipeline Usage Guide

## Introduction

`LayerDPipeline` is the high-level API for LayerD that provides a complete, streamlined workflow from image input to export. It orchestrates all the steps needed for layer decomposition:

1. **Decomposition** - Extract layers using BiRefNet-based matting
2. **Organization** - Group pixels into semantic elements
3. **Classification** - Identify element types (text/vector/image)
4. **Export** - Generate SVG or PSD output

### When to Use LayerDPipeline

Use `LayerDPipeline` when you want:
- A complete end-to-end workflow with minimal code
- Organized elements with type classification
- Direct export to SVG or PSD formats
- Sensible defaults with easy customization

Use the low-level `LayerD` class when you need:
- Fine-grained control over each processing step
- Custom postprocessing beyond the standard pipeline
- Integration with existing layer processing code

See the [Inference Guide](inference.md) for detailed low-level API documentation.

### Quick Comparison

```python
# High-level pipeline (recommended for most use cases)
from layerd import LayerDPipeline
pipeline = LayerDPipeline()
result = pipeline(image)
result.save("output.svg")

# Low-level API (for custom workflows)
from layerd import LayerD
layerd = LayerD(matting_hf_card="cyberagent/layerd-birefnet")
layers = layerd.decompose(image)
# ... custom postprocessing ...
```

## Installation

Install LayerD from the GitHub repository:

```bash
pip install git+https://github.com/CyberAgentAILab/LayerD.git
```

This includes SVG and PSD export support. For additional features:

```bash
# With training dependencies
pip install "git+https://github.com/CyberAgentAILab/LayerD.git#egg=layerd[train]"

# With dataset generation
pip install "git+https://github.com/CyberAgentAILab/LayerD.git#egg=layerd[dataset]"

# All optional dependencies
pip install "git+https://github.com/CyberAgentAILab/LayerD.git#egg=layerd[all]"
```

See the [Installation Guide](installation.md) for more options.

## Quick Start

```python
from layerd import LayerDPipeline
from PIL import Image

# Create pipeline
pipeline = LayerDPipeline(device="cpu")

# Load and process image
image = Image.open("design.png")
result = pipeline(image)

# Export to SVG (auto-detects format from extension)
result.save("output.svg")
```

That's it! The pipeline handles decomposition, organization, and export in one call.

## Basic Usage

### Creating a Pipeline

```python
from layerd import LayerDPipeline

# Basic pipeline with defaults
pipeline = LayerDPipeline()

# With device specification
pipeline = LayerDPipeline(device="cuda")

# With custom configuration
pipeline = LayerDPipeline(
    matting_hf_card="cyberagent/layerd-birefnet",
    device="cuda",
    overlap_threshold=0.9,
    labeler_threshold=5.0
)
```

### Processing Images

```python
from PIL import Image

# Load image
image = Image.open("design.png")

# Process with default settings
result = pipeline(image)

# Process with custom iterations
result = pipeline(image, max_iterations=5)
```

### Accessing Results

The `PipelineResult` object contains:

```python
# Organized elements with classification
print(f"Found {len(result.elements)} elements")
for elem in result.elements:
    print(f"Element {elem['id']}: type={elem['type']}, box={elem['box']}")
    # elem['image'] is the cropped RGBA PIL Image
    # elem['box'] contains x_min, y_min, x_max, y_max

# Raw RGBA layers
print(f"Decomposed into {len(result.layers)} layers")
for i, layer in enumerate(result.layers):
    layer.save(f"layer_{i}.png")

# Canvas size
width, height = result.canvas_size
print(f"Original size: {width}x{height}")

# OCR result (future feature - currently None)
if result.ocr_result:
    print("OCR data:", result.ocr_result)
```

### Exporting Results

```python
# Auto-detect format from extension
result.save("output.svg")  # SVG export
result.save("output.psd")  # PSD export

# Get export data as string/bytes
svg_string = result.to_svg()
psd_bytes = result.to_psd()

# SVG with external images
result.save("output.svg", image_mode="external", image_dir="./images")
```

See the [Export Guide](export.md) for detailed export documentation.

## Configuration

### Matting Parameters

Control the BiRefNet matting model:

```python
pipeline = LayerDPipeline(
    # Model selection
    matting_hf_card="cyberagent/layerd-birefnet",  # HuggingFace model card
    matting_process_size=(1024, 1024),              # Processing resolution (None = auto)
    matting_weight_path="/path/to/weights.pth",     # Custom weights (overrides HF card)
)
```

### Refinement Options

Control color refinement and unblending:

```python
pipeline = LayerDPipeline(
    use_unblend=True,           # Enable color unblending (default: True)
    bg_refine=True,             # Refine background colors (default: True)
    fg_refine=True,             # Refine foreground colors (default: True)
    fg_refine_num_colors=2,     # Number of foreground colors (default: 2)
    bg_refine_num_colors=10,    # Number of background colors (default: 10)
    kernel_scale=0.015,         # Refinement kernel scale (default: 0.015)
)
```

### Organization Parameters

Control how layers are organized into elements:

```python
pipeline = LayerDPipeline(
    overlap_threshold=0.9,      # Overlap threshold for element merging (default: 0.9)
                                # Higher = stricter separation
)
```

### Classification

Control element type classification:

```python
# Disable classification (all elements marked as "image")
pipeline = LayerDPipeline(labeler=None)

# Use default EntropyLabeler with custom threshold
pipeline = LayerDPipeline(labeler_threshold=5.0)  # Default: 5.0
                                                   # Lower = more elements marked as "vector"

# Use advanced gradient-aware labeler
from layerd import GradientAwareLabeler
labeler = GradientAwareLabeler(
    entropy_threshold=5.0,
    gradient_weight=0.3
)
pipeline = LayerDPipeline(labeler=labeler)
```

### Device Management

```python
# Set device at initialization
pipeline = LayerDPipeline(device="cuda")

# Change device later (returns self for chaining)
pipeline = pipeline.to("cpu")

# GPU with specific CUDA device
pipeline = LayerDPipeline(device="cuda:1")
```

## Advanced Features

### Custom Element Classification

Implement your own labeler by subclassing `ElementLabeler`:

```python
from layerd import ElementLabeler
from layerd.types import Element

class MyCustomLabeler(ElementLabeler):
    def label(self, element: Element) -> str:
        """Classify element as 'text', 'vector', or 'image'."""
        # Custom classification logic
        image = element["image"]
        if self.is_text(image):
            return "text"
        elif self.is_vector(image):
            return "vector"
        else:
            return "image"

    def is_text(self, image):
        # Your text detection logic
        pass

    def is_vector(self, image):
        # Your vector detection logic
        pass

# Use custom labeler
pipeline = LayerDPipeline(labeler=MyCustomLabeler())
```

### Custom Matting Weights

Load weights from local paths or cloud storage:

```python
# Local weights
pipeline = LayerDPipeline(matting_weight_path="./my_birefnet.pth")

# Remote weights (requires appropriate fsspec backend, e.g., gcsfs for gs://)
pipeline = LayerDPipeline(matting_weight_path="gs://my-bucket/models/birefnet.pth")
```

### Element Structure

Each element in `result.elements` is a `TypedDict` with:

```python
{
    "id": 1,                          # Unique element ID
    "type": "text",                   # Element type: "text", "vector", or "image"
    "image": <PIL.Image>,             # Cropped RGBA image
    "box": {                          # Bounding box coordinates
        "x_min": 10,                  # Left edge (inclusive)
        "y_min": 20,                  # Top edge (inclusive)
        "x_max": 110,                 # Right edge (exclusive)
        "y_max": 70                   # Bottom edge (exclusive)
    }
}
```

### Batch Processing

Process multiple images efficiently:

```python
from pathlib import Path

# Create pipeline once
pipeline = LayerDPipeline(device="cuda")

# Process multiple images
image_dir = Path("./designs")
for image_path in image_dir.glob("*.png"):
    image = Image.open(image_path)
    result = pipeline(image)

    output_path = f"./output/{image_path.stem}.svg"
    result.save(output_path)

    print(f"Processed {image_path.name}: {len(result.elements)} elements")
```

## API Reference

### LayerDPipeline

```python
class LayerDPipeline:
    def __init__(
        self,
        # Matting parameters
        matting_hf_card: str = "cyberagent/layerd-birefnet",
        matting_process_size: tuple[int, int] | None = None,
        matting_weight_path: str | None = None,

        # Refinement parameters
        use_unblend: bool = True,
        bg_refine: bool = True,
        fg_refine: bool = True,
        fg_refine_num_colors: int = 2,
        bg_refine_num_colors: int = 10,
        kernel_scale: float = 0.015,

        # Organization parameters
        overlap_threshold: float = 0.9,

        # Classification parameters
        labeler: ElementLabeler | None = <default EntropyLabeler>,
        labeler_threshold: float = 5.0,  # Only used if labeler not provided

        # Device
        device: str = "cpu",
    ) -> None

    def __call__(
        self,
        image: Image.Image,
        max_iterations: int = 3,
    ) -> PipelineResult

    def to(self, device: str) -> LayerDPipeline
```

### PipelineResult

```python
class PipelineResult:
    elements: list[Element]           # Organized elements with classification
    layers: list[Image.Image]         # Raw RGBA layers
    ocr_result: dict | None           # OCR result (future feature, currently None)
    canvas_size: tuple[int, int]      # Original image dimensions

    def to_svg(
        self,
        image_mode: Literal["base64", "external"] = "base64",
        image_dir: str | None = None,
    ) -> str

    def to_psd(self) -> bytes

    def save(
        self,
        path: str,
        format: str | None = None,  # Auto-detected from extension
        **kwargs: Any,               # Format-specific options
    ) -> None
```

### Element Type

```python
class Element(TypedDict):
    id: int                  # Unique element identifier
    type: str                # "text", "vector", or "image"
    image: Image.Image       # Cropped RGBA image
    box: BoundingBox         # Bounding box coordinates

class BoundingBox(TypedDict):
    x_min: int              # Left edge (inclusive)
    y_min: int              # Top edge (inclusive)
    x_max: int              # Right edge (exclusive)
    y_max: int              # Bottom edge (exclusive)
```

## Examples

### Basic Decomposition

```python
from layerd import LayerDPipeline
from PIL import Image

pipeline = LayerDPipeline(device="cuda")
image = Image.open("design.png")

result = pipeline(image, max_iterations=3)
result.save("output.svg")

print(f"Decomposed into {len(result.elements)} elements")
```

### Custom Classification

```python
from layerd import LayerDPipeline, GradientAwareLabeler

# Use gradient-aware labeler for better classification
labeler = GradientAwareLabeler(entropy_threshold=4.5)
pipeline = LayerDPipeline(labeler=labeler, device="cuda")

result = pipeline(image)

# Print classification results
for elem in result.elements:
    print(f"Element {elem['id']}: {elem['type']}")
```

### Multiple Export Formats

```python
result = pipeline(image)

# Export to both formats
result.save("output.svg")
result.save("output.psd")

# Or get the data directly
svg_string = result.to_svg()
psd_bytes = result.to_psd()

with open("manual.svg", "w") as f:
    f.write(svg_string)
```

### Processing with Custom Weights

```python
# Train custom BiRefNet model (see training.md)
# Then use it in the pipeline
pipeline = LayerDPipeline(
    matting_weight_path="./checkpoints/my_birefnet.pth",
    device="cuda"
)

result = pipeline(image)
```

## Future Features

### OCR Integration

OCR support is planned for a future release (see [GitHub Issue #86](https://github.com/CyberAgentAILab/LayerD/issues/86)). The pipeline already has an `ocr_result` field reserved for this feature:

```python
result = pipeline(image)

# Currently None, will contain OCR data in future release
if result.ocr_result:
    for block in result.ocr_result["blocks"]:
        print(f"Text: {block['text']}")
```

The OCR integration will enable:
- Text detection and recognition
- Automatic text element classification
- Text layer organization
- OCR-guided element extraction

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

```python
# Reduce processing size
pipeline = LayerDPipeline(
    matting_process_size=(512, 512),  # Smaller than default
    device="cuda"
)

# Or reduce iterations
result = pipeline(image, max_iterations=2)

# Or use CPU
pipeline = LayerDPipeline(device="cpu")
```

### Poor Element Classification

If elements are misclassified:

```python
# Adjust labeler threshold
pipeline = LayerDPipeline(labeler_threshold=4.0)  # Lower = more vectors

# Or use gradient-aware labeler
from layerd import GradientAwareLabeler
pipeline = LayerDPipeline(
    labeler=GradientAwareLabeler(entropy_threshold=4.0)
)

# Or disable classification
pipeline = LayerDPipeline(labeler=None)
```

### Slow Processing

For faster processing:

```python
# Use GPU
pipeline = LayerDPipeline(device="cuda")

# Reduce iterations
result = pipeline(image, max_iterations=2)

# Disable refinement (faster but lower quality)
pipeline = LayerDPipeline(
    bg_refine=False,
    fg_refine=False,
    use_unblend=False
)
```

### Import Errors

If you get import errors:

```bash
# Ensure LayerD is installed
pip install git+https://github.com/CyberAgentAILab/LayerD.git

# Check Python version (requires >=3.10)
python --version

# Check torch installation
python -c "import torch; print(torch.__version__)"
```

## See Also

- [Export Guide](export.md) - Detailed SVG/PSD export documentation
- [Architecture](architecture.md) - System architecture and design patterns
- [Training Guide](training.md) - Training custom matting models
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
