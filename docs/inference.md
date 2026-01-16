# Inference Guide

This guide covers the low-level `LayerD` API for layer decomposition. For complete end-to-end workflows with export support, see the [Pipeline Guide](pipeline.md).

> **Note**: For most users, we recommend using [`LayerDPipeline`](pipeline.md) which provides a complete workflow including decomposition, organization, classification, and export to SVG/PSD formats.

## Quick Start

The simplest way to decompose an image:

```python
from PIL import Image
from layerd import LayerD

# Load your image
image = Image.open("your_image.png")

# Create LayerD instance
layerd = LayerD(matting_hf_card="cyberagent/layerd-birefnet").to("cpu")

# Decompose into layers
layers = layerd.decompose(image)

# Save layers
for i, layer in enumerate(layers):
    layer.save(f"layer_{i}.png")
```

## Python API

### Basic Usage

```python
from layerd import LayerD

# Initialize with default settings
layerd = LayerD(
    matting_hf_card="cyberagent/layerd-birefnet",  # HuggingFace model card
).to("cpu")  # or "cuda" for GPU

# Decompose image
layers = layerd.decompose(image, max_iterations=3)
```

### Advanced Options

```python
layerd = LayerD(
    matting_hf_card="cyberagent/layerd-birefnet",
    matting_process_size=(1024, 1024),  # Process size for matting model
    kernel_scale=0.015,                  # Mask expansion scale (default: 0.015)
    use_unblend=True,                    # Use unblending for color estimation
    fg_refine=True,                      # Refine foreground layers
    bg_refine=True,                      # Refine background
)

# Move to GPU for faster inference
layerd = layerd.to("cuda")

# Decompose with custom iterations
layers = layerd.decompose(image, max_iterations=5)
```

### Parameters

#### LayerD Initialization

- **`matting_hf_card`** (str): HuggingFace model card for matting model
  - Default: `"cyberagent/layerd-birefnet"`
  - Use this to specify different matting models

- **`matting_process_size`** (tuple[int, int] | None): Process size for matting model as (width, height)
  - Default: `None` (uses model's default size)
  - Smaller sizes = faster but lower quality
  - Larger sizes = slower but higher quality
  - Example: `(512, 512)` or `(1024, 1024)`

- **`kernel_scale`** (float): Scale factor for mask expansion
  - Default: `0.015`
  - Higher values = more expansion (helps with anti-aliased edges)
  - Lower values = less expansion (sharper but may miss edge details)
  - Calculated as: `kernel_size = int(min(H, W) * kernel_scale)`

- **`use_unblend`** (bool): Estimate foreground color by unblending
  - Default: `True`
  - Helps separate foreground from background colors

- **`fg_refine`** (bool): Refine foreground alpha and colors
  - Default: `True`
  - Uses flat color region detection for better quality

- **`bg_refine`** (bool): Refine background with palette-based color assignment
  - Default: `True`
  - Improves background quality

#### decompose() Method

- **`image`** (PIL.Image.Image): Input image in RGB or RGBA format
- **`max_iterations`** (int): Maximum number of decomposition iterations
  - Default: `3`
  - More iterations may extract more layers but increase processing time
  - Decomposition stops early if no more layers detected

### Output Format

The `decompose()` method returns a list of PIL Images in RGBA format:

```python
layers = layerd.decompose(image)
# layers[0] = background layer
# layers[1] = topmost foreground layer
# layers[2] = second foreground layer (if exists)
# ...
```

**Layer order**: `[background, topmost_fg, ..., bottommost_fg]`

Each layer is a PIL Image with:

- Mode: RGBA
- Size: Same as input image
- Alpha channel: Represents layer opacity/mask

## Command-Line Interface (CLI)

### Basic Usage

```bash
# Decompose a single image
layerd --input image.png --output-dir outputs/

# Use GPU for faster processing
layerd --input image.png --output-dir outputs/ --device cuda

# Specify maximum iterations
layerd --input image.png --output-dir outputs/ --max-iterations 5
```

### CLI Options

- **`--input`** (required): Path to input image file
- **`--output-dir`** (required): Output directory to save results
- **`--device`**: Device to run on (`cpu` or `cuda`, default: `cpu`)
- **`--max-iterations`**: Maximum decomposition iterations (default: `3`)
- **`--matting-hf-card`**: HuggingFace model card (default: `cyberagent/layerd-birefnet`)
- **`--matting-process-size`**: Process size as width and height (e.g., `512 512`)
- **`--log-level`**: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, default: `INFO`)

### CLI Examples

```bash
# Basic single file inference
layerd --input path/to/image.png --output-dir outputs/

# With custom device and iterations
layerd --input image.png --output-dir outputs/ --device cuda --max-iterations 5

# With custom matting model size
layerd --input image.png --output-dir outputs/ --matting-process-size 512 512

# With debug logging
layerd --input image.png --output-dir outputs/ --log-level DEBUG
```

### CLI Output

The CLI saves outputs to the specified directory:

```
outputs/
├── layer_0.png  # Background
├── layer_1.png  # Topmost foreground
├── layer_2.png  # Second foreground (if exists)
└── ...
```

## Batch Inference

For processing multiple images, directories, or glob patterns, use the batch inference script:

### Basic Batch Processing

```bash
# Process a directory
uv run python ./tools/infer.py \
  --input /path/to/images/ \
  --output-dir outputs/ \
  --device cuda

# Process with glob pattern
uv run python ./tools/infer.py \
  --input "data/*.png" \
  --output-dir outputs/ \
  --device cuda

# Process multiple specific files
uv run python ./tools/infer.py \
  --input image1.png image2.png image3.png \
  --output-dir outputs/ \
  --device cpu
```

### Batch Script Options

The batch inference script supports all the same options as the CLI, plus:

- **`--input`**: Can be file, directory, glob pattern, or list of files
- **`--matting-weight-path`**: Path to custom trained matting model weights
  - If not specified, uses default model from HuggingFace

### Custom Model Weights

To use custom trained weights:

```bash
uv run python ./tools/infer.py \
  --input "data/*.png" \
  --output-dir outputs/ \
  --matting-weight-path /path/to/trained/weights.pth \
  --device cuda
```

> **Note**: For using custom weights with the high-level pipeline API, see [Pipeline Guide - Custom Weights](pipeline.md#custom-weights).

## Input Recommendations

### Image Format

- **Recommended**: PNG format
  - Lossless compression preserves edge quality
  - Important for text and sharp graphics
- **Avoid**: JPEG format
  - Compression artifacts around edges
  - Can degrade inpainting quality, especially around text

### Image Quality

- Use high-resolution images when possible
- Avoid pre-processed or heavily compressed images
- Clean, well-lit images work best

### Supported Formats

LayerD accepts any format supported by PIL:

- PNG (recommended)
- JPEG
- BMP
- TIFF
- WebP

## Performance Tips

### Speed Optimization

1. **Use GPU**: Significantly faster than CPU

   ```python
   layerd = layerd.to("cuda")
   ```

2. **Reduce process size**: Faster but lower quality

   ```python
   layerd = LayerD(matting_process_size=(512, 512))
   ```

3. **Reduce iterations**: Stop early if layers are satisfactory

   ```python
   layers = layerd.decompose(image, max_iterations=2)
   ```

### Quality Optimization

1. **Use PNG input**: Avoid compression artifacts

2. **Increase process size**: Slower but higher quality

   ```python
   layerd = LayerD(matting_process_size=(1024, 1024))
   ```

3. **Adjust kernel_scale**: Fine-tune edge handling

   ```python
   # For sharper edges (graphics with clean lines)
   layerd = LayerD(kernel_scale=0.010)

   # For softer edges (photos with gradual transitions)
   layerd = LayerD(kernel_scale=0.020)
   ```

### Memory Management

For large images or limited memory:

```python
# Use smaller process size
layerd = LayerD(matting_process_size=(512, 512))

# Process on CPU if GPU memory is limited
layerd = layerd.to("cpu")

# Process images one at a time in batch inference
# (batch script does this automatically)
```

## Troubleshooting

### Common Issues

**Problem**: "CUDA out of memory"

**Solution**: Reduce process size or use CPU:

```python
layerd = LayerD(matting_process_size=(512, 512))
# or
layerd = layerd.to("cpu")
```

**Problem**: Poor quality around text edges

**Solution**: Use PNG input and adjust kernel_scale:

```python
layerd = LayerD(kernel_scale=0.020)  # Increase for better edge handling
```

**Problem**: Too few or too many layers extracted

**Solution**: Adjust max_iterations:

```python
# Extract more layers
layers = layerd.decompose(image, max_iterations=5)

# Extract fewer layers
layers = layerd.decompose(image, max_iterations=2)
```

For more troubleshooting help, see [troubleshooting.md](troubleshooting.md).

## Examples

### Example 1: Simple Decomposition

```python
from PIL import Image
from layerd import LayerD

image = Image.open("design.png")
layerd = LayerD(matting_hf_card="cyberagent/layerd-birefnet").to("cpu")
layers = layerd.decompose(image)

for i, layer in enumerate(layers):
    layer.save(f"output/layer_{i}.png")
print(f"Extracted {len(layers)} layers")
```

### Example 2: Batch Processing Script

```python
from pathlib import Path
from PIL import Image
from layerd import LayerD

input_dir = Path("inputs/")
output_dir = Path("outputs/")
output_dir.mkdir(exist_ok=True)

layerd = LayerD(matting_hf_card="cyberagent/layerd-birefnet").to("cuda")

for img_path in input_dir.glob("*.png"):
    print(f"Processing {img_path.name}")
    image = Image.open(img_path)
    layers = layerd.decompose(image)

    # Save layers in subdirectory
    img_output_dir = output_dir / img_path.stem
    img_output_dir.mkdir(exist_ok=True)

    for i, layer in enumerate(layers):
        layer.save(img_output_dir / f"layer_{i}.png")
```

### Example 3: Custom Configuration

```python
from PIL import Image
from layerd import LayerD

image = Image.open("complex_design.png")

layerd = LayerD(
    matting_hf_card="cyberagent/layerd-birefnet",
    matting_process_size=(1024, 1024),  # High quality
    kernel_scale=0.020,                  # Better edge handling
    use_unblend=True,
    fg_refine=True,
    bg_refine=True,
).to("cuda")

layers = layerd.decompose(image, max_iterations=5)

for i, layer in enumerate(layers):
    layer.save(f"output/layer_{i}.png")
```

## Related Documentation

- [Installation Guide](installation.md) - Setup instructions
- [Training Guide](training.md) - Train custom matting models
- [Architecture](architecture.md) - Understanding the pipeline
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
