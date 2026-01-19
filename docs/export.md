# Export Format Documentation

## Overview

LayerD supports exporting decomposed layers to two formats:

- **SVG** - Scalable Vector Graphics with embedded or external images
- **PSD** - Adobe Photoshop Document with editable layers

Both formats preserve the layer structure and can be edited in design tools. Export is performed through `PipelineResult` methods after running the pipeline.

```python
from layerd import LayerDPipeline

pipeline = LayerDPipeline()
result = pipeline(image)

# Export to either format
result.save("output.svg")  # SVG
result.save("output.psd")  # PSD
```

## SVG Export

### Basic Usage

SVG export creates a standards-compliant SVG file with images embedded as data URIs (default) or external file references:

```python
# Default: base64 embedded images
result.save("output.svg")

# Or get the SVG string
svg_string = result.to_svg()
with open("output.svg", "w") as f:
    f.write(svg_string)
```

### Image Embedding Modes

#### Base64 Mode (Default)

Images are embedded as data URIs directly in the SVG:

```python
result.save("output.svg")  # Uses base64 by default

# Explicit base64 mode
svg_string = result.to_svg(image_mode="base64")
```

**Advantages:**

- Single self-contained file
- No external dependencies
- Easy to share and view

**Disadvantages:**

- Larger file size (~33% larger than binary)
- Not suitable for very large images

#### External Mode

Images are saved as separate PNG files and referenced by path:

```python
# Save with external images
result.save("output.svg", image_mode="external", image_dir="./images")

# Or using to_svg()
svg_string = result.to_svg(image_mode="external", image_dir="./images")
with open("output.svg", "w") as f:
    f.write(svg_string)
```

This creates:

```
./output.svg
./images/element_0.png
./images/element_1.png
./images/element_2.png
...
```

**Advantages:**

- Smaller SVG file size
- Images can be edited separately
- Better for version control (text diff-able SVG)

**Disadvantages:**

- Multiple files to manage
- Requires preserving directory structure

### SVG Structure

The generated SVG has a clean structure with metadata:

```xml
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <image x="10" y="20" width="100" height="50"
         href="data:image/png;base64,..."
         data-type="text"
         data-id="0" />
  <image x="150" y="80" width="200" height="150"
         href="data:image/png;base64,..."
         data-type="vector"
         data-id="1" />
  ...
</svg>
```

**Metadata Attributes:**

- `data-type`: Element type ("text", "vector", or "image")
- `data-id`: Unique element identifier

These attributes enable round-trip conversion and programmatic manipulation.

### Compatibility

SVG files can be opened in:

- **Web Browsers** - Chrome, Firefox, Safari, Edge (native support)
- **Vector Editors** - Adobe Illustrator, Inkscape, Figma
- **Image Viewers** - Most modern image viewers
- **Design Tools** - Sketch, Affinity Designer

## PSD Export

### Basic Usage

PSD export creates an Adobe Photoshop document with each element as a separate layer:

```python
# Export to PSD
result.save("output.psd")

# Or get PSD bytes
psd_bytes = result.to_psd()
with open("output.psd", "wb") as f:
    f.write(psd_bytes)
```

### Layer Structure

In Photoshop, you'll see:

- Each element as a separate layer
- Layers named by element ID and type (e.g., "element_0_text")
- Original positioning preserved
- RGBA transparency maintained

### File Size Considerations

PSD files are typically **larger than SVG** because they store full raster data:

- SVG with base64: ~33% larger than PNG
- SVG with external: Small text file + PNG images
- PSD: Similar to sum of all PNGs + overhead

For a design with 10 elements:

- SVG (base64): ~500 KB
- SVG (external): ~50 KB SVG + 450 KB images
- PSD: ~600 KB

### Compatibility

PSD files can be opened in:

- **Adobe Photoshop** - Native format
- **GIMP** - Open source image editor
- **Affinity Photo** - Professional photo editor
- **Photopea** - Web-based Photoshop alternative

## Format Comparison

| Feature | SVG | PSD |
| ------- | --- | --- |
| File size | Small-Medium | Large |
| Web display | Native browser support | Requires conversion |
| Editing tools | Browsers, Illustrator, Figma | Photoshop, GIMP |
| Text preservation | As raster images | As raster images |
| Layer metadata | Yes (data attributes) | Limited |
| Scalability | Vector format (scalable) | Raster format (fixed resolution) |
| Transparency | Full RGBA support | Full RGBA support |
| Single file | Optional (base64 mode) | Yes |
| Version control | Good (especially external mode) | Poor (binary format) |

## Choosing a Format

### Use SVG When

- You need web compatibility
- File size is important
- You want version control-friendly output
- You'll edit in vector tools (Illustrator, Figma)
- You want programmatic manipulation

### Use PSD When

- You're working in Photoshop
- You need native Adobe ecosystem support
- You want full raster editing capabilities
- Single-file distribution is important

### Use Both

Export to both formats for maximum flexibility:

```python
result.save("output.svg")
result.save("output.psd")
```

## Using the save() Method

The `save()` method auto-detects format from file extension:

```python
# Auto-detection
result.save("output.svg")  # Calls to_svg()
result.save("output.psd")  # Calls to_psd()

# Explicit format (overrides extension)
result.save("output.dat", format="svg")

# Format-specific options
result.save("output.svg", image_mode="external", image_dir="./images")
```

## Advanced Usage

### External Image Workflow

For large projects with many images:

```python
from pathlib import Path

# Process multiple designs
designs_dir = Path("./designs")
output_dir = Path("./output")

pipeline = LayerDPipeline(device="cuda")

for design_path in designs_dir.glob("*.png"):
    image = Image.open(design_path)
    result = pipeline(image)

    # Create dedicated image directory per design
    design_name = design_path.stem
    svg_path = output_dir / f"{design_name}.svg"
    img_dir = output_dir / f"{design_name}_images"

    result.save(str(svg_path), image_mode="external", image_dir=str(img_dir))
```

This creates:

```
./output/
  design1.svg
  design1_images/
    element_0.png
    element_1.png
  design2.svg
  design2_images/
    element_0.png
    element_1.png
```

### Batch Export to Multiple Formats

Export the same decomposition to multiple formats:

```python
result = pipeline(image)

# Export to all formats
formats = {
    "svg_base64": ("output_base64.svg", {"image_mode": "base64"}),
    "svg_external": ("output_external.svg", {"image_mode": "external", "image_dir": "./images"}),
    "psd": ("output.psd", {}),
}

for name, (path, kwargs) in formats.items():
    result.save(path, **kwargs)
    print(f"Saved {name}: {path}")
```

### Custom File Organization

Organize exported files by element type:

```python
result = pipeline(image)

# Group elements by type
from collections import defaultdict
by_type = defaultdict(list)
for elem in result.elements:
    by_type[elem["type"]].append(elem)

# Export each type separately
for elem_type, elements in by_type.items():
    # Create custom PipelineResult with filtered elements
    filtered_result = PipelineResult(
        elements=elements,
        layers=result.layers,
        ocr_result=result.ocr_result,
        canvas_size=result.canvas_size
    )
    filtered_result.save(f"output_{elem_type}.svg")
```

## API Reference

### PipelineResult.to_svg()

```python
def to_svg(
    self,
    image_mode: Literal["base64", "external"] = "base64",
    image_dir: str | None = None,
) -> str
```

Generate SVG string representation.

**Parameters:**

- `image_mode`: Image embedding mode
  - `"base64"` (default): Embed images as data URIs
  - `"external"`: Save images to directory and reference by path
- `image_dir`: Directory for external images (required if `image_mode="external"`)

**Returns:**

- SVG string

**Raises:**

- `ValueError`: If `image_mode="external"` but `image_dir` not provided

### PipelineResult.to_psd()

```python
def to_psd(self) -> bytes
```

Generate PSD bytes representation.

**Returns:**

- PSD file as bytes

### PipelineResult.save()

```python
def save(
    self,
    path: str,
    format: str | None = None,
    **kwargs: Any,
) -> None
```

Save result to file with format auto-detection.

**Parameters:**

- `path`: Output file path (supports various file systems via `fsspec`: local paths, `gs://`, `s3://`, `abfs://`, `https://`)
- `format`: Export format (`"svg"` or `"psd"`), auto-detected from extension if `None`
- `**kwargs`: Format-specific options passed to `to_svg()` or `to_psd()`

**Raises:**

- `ValueError`: If format cannot be determined or is unsupported

**Note:** This method supports various file systems through `fsspec`, allowing you to save directly to cloud storage (e.g., Google Cloud Storage, Amazon S3, Azure Blob Storage) or HTTP endpoints.

## Examples

### SVG with Base64 Images

```python
from layerd import LayerDPipeline
from PIL import Image

pipeline = LayerDPipeline()
image = Image.open("design.png")
result = pipeline(image)

# Simple save (default base64)
result.save("output.svg")

# Explicit base64
svg_string = result.to_svg(image_mode="base64")
with open("output_explicit.svg", "w") as f:
    f.write(svg_string)
```

### SVG with External Images

```python
# Save with external images
result.save("output.svg", image_mode="external", image_dir="./images")

# Verify files
from pathlib import Path
print(f"SVG size: {Path('output.svg').stat().st_size} bytes")
print(f"Images: {list(Path('./images').glob('*.png'))}")
```

### PSD Export

```python
# Simple PSD export
result.save("output.psd")

# Or with explicit bytes
psd_bytes = result.to_psd()
with open("output_manual.psd", "wb") as f:
    f.write(psd_bytes)

print(f"PSD size: {len(psd_bytes)} bytes")
```

### Comparing Formats

```python
import os

result = pipeline(image)

# Export to all formats
result.save("output_base64.svg", image_mode="base64")
result.save("output_external.svg", image_mode="external", image_dir="./images")
result.save("output.psd")

# Compare file sizes
formats = ["output_base64.svg", "output_external.svg", "output.psd"]
for fmt in formats:
    size = os.path.getsize(fmt)
    print(f"{fmt}: {size:,} bytes ({size/1024:.1f} KB)")
```

## See Also

- [Pipeline Guide](pipeline.md) - LayerDPipeline usage and configuration
- [Architecture](architecture.md) - Export module architecture
- [Troubleshooting](troubleshooting.md) - Common export issues
