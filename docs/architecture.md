# Architecture

This document describes the internal architecture and design patterns of LayerD.

## Overview

LayerD is a layer decomposition method that extracts editable layers from raster graphic design images. The system uses a two-stage iterative approach:

1. **Top-layer matting**: Extracts the alpha matte of the topmost layer using BiRefNet
2. **Inpainting**: Fills in the removed content using LaMa to reconstruct the background

The main `LayerD` class orchestrates this pipeline iteratively to decompose an image into multiple layers (background + foreground layers).

## Core Components

### Main Pipeline

The main pipeline is implemented in [src/layerd/models/layerd.py](../src/layerd/models/layerd.py):

- **`LayerD` class**: Main interface for layer decomposition
- **`decompose()`**: Iteratively extracts layers (max 3 iterations by default)
- **`_decompose_step()`**: Single iteration of matting + inpainting
- Uses helper functions from `helpers.py` for refinement operations

### Model Abstraction

LayerD uses a registry pattern to support multiple model backends:

- **Base classes**: `BaseMatting` and `BaseInpaint` define model interfaces
- **Registry pattern**: Implemented in `models/matting/__init__.py` and `models/inpaint/__init__.py`
- **Factory functions**: Use `build_matting()` and `build_inpaint()` to instantiate models
- **Current implementations**:
  - Matting: BiRefNet (HuggingFace model)
  - Inpainting: LaMa (from simple-lama-inpainting)

### Refinement Pipeline

The decomposition includes optional refinement steps controlled by flags:

- **`use_unblend`**: Estimates foreground color by unblending (subtracting background)
- **`fg_refine`**: Refines foreground alpha and colors using flat color region detection
- **`bg_refine`**: Refines background with palette-based color assignment

These refinement steps help improve the quality of the extracted layers, especially for text and flat-color graphics.

### Evaluation System

Evaluation components are in [src/layerd/evaluation/](../src/layerd/evaluation/):

- **`LayersEditDist`**: Main metric for layer decomposition quality
- **Dynamic Time Warping (DTW)**: Aligns predicted and ground truth layers
- **Edit distance**: Computes edit operations (insert, delete, modify) between layer sequences
- **Per-layer metrics**: RGBL1 (color accuracy), AlphaIoU (alpha mask accuracy)

See [evaluation.md](evaluation.md) for usage details.

## Module Organization

```
src/layerd/
├── models/
│   ├── layerd.py          # Main LayerD class
│   ├── helpers.py         # Refinement utilities (unblend, mask ops, color estimation)
│   ├── matting/           # Matting model implementations
│   │   ├── base.py        # BaseMatting abstract class
│   │   ├── birefnet_matting.py
│   │   └── __init__.py    # Registry with build_matting()
│   └── inpaint/           # Inpainting model implementations
│       ├── base.py        # BaseInpaint abstract class
│       ├── lama_inpaint.py
│       └── __init__.py    # Registry with build_inpaint()
├── matting/birefnet/      # BiRefNet training code
│   ├── train.py           # Training loop
│   ├── dataset.py         # Dataset implementation
│   ├── loss.py            # Loss functions
│   └── image_proc.py      # Image preprocessing
├── data/                  # Dataset utilities
│   ├── crello.py          # Crello dataset handling
│   └── renderer.py        # Rendering utilities
├── evaluation/            # Evaluation metrics
│   ├── edit_distance.py   # LayersEditDist metric
│   ├── dtw.py             # Dynamic Time Warping
│   ├── edits.py           # Edit operations
│   └── metrics.py         # Per-layer metrics (RGBL1, AlphaIoU)
├── configs/               # Hydra configuration files
│   └── train.yaml         # Training hyperparameters
└── _vendor/               # Bundled dependencies (see below)
    ├── simple_lama_inpainting/
    └── cr_renderer/
```

## Key Design Patterns

1. **Factory Pattern**: Models are created via `build_matting()` and `build_inpaint()` functions with string identifiers
2. **Abstract Base Classes**: All models inherit from `BaseMatting` or `BaseInpaint` with validation
3. **Iterative Decomposition**: `decompose()` runs `_decompose_step()` until no more layers or max iterations reached
4. **PIL Image Interface**: Main API uses PIL Images; internal processing uses numpy arrays

## Important Implementation Details

### Input and Output Formats

- **Input Requirements**: Prefer PNG images to avoid compression artifacts around text edges
- **PIL Image Interface**: The main API uses PIL Images in RGBA format
- **Internal Processing**: Uses numpy arrays (float64) for computation
- **Alpha Format**: Matting models output float64 alpha in [0, 1] range

### Layer Extraction Process

- **Mask Expansion**: Uses `kernel_scale` parameter (default 0.015) to expand masks based on image dimensions
  - This helps capture anti-aliased edges properly
  - Scale is relative to image size: `kernel_size = int(min(H, W) * kernel_scale)`
- **Layer Order**: `decompose()` returns `[background, topmost_fg, ..., bottommost_fg]`
  - The background is always the first layer
  - Subsequent layers are ordered from top to bottom as they were extracted

### Model Loading

- **First run**: Downloads models from remote sources
  - BiRefNet: ~1GB from HuggingFace ([cyberagent/layerd-birefnet](https://huggingface.co/cyberagent/layerd-birefnet))
  - LaMa: ~200MB from GitHub ([simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting))
- **Caching**: Models are cached locally for subsequent runs
- **Device placement**: Models can run on CPU or CUDA devices

## Bundled Dependencies

LayerD bundles two dependencies under `layerd._vendor` to enable numpy 2.0 compatibility:

### 1. simple-lama-inpainting (`layerd._vendor.simple_lama_inpainting`)

- **Original**: <https://github.com/enesmsahin/simple-lama-inpainting>
- **PyPI**: <https://pypi.org/project/simple-lama-inpainting/> (outdated, numpy 1.x)
- **Purpose**: LaMa inpainting model wrapper
- **License**: Apache-2.0
- **Reason for bundling**: PyPI version uses numpy 1.x (incompatible with LayerD's numpy 2.0 requirement)

### 2. cr-renderer (`layerd._vendor.cr_renderer`)

- **Original**: <https://github.com/CyberAgentAILab/cr-renderer>
- **Revision**: a17e1fb
- **Purpose**: Crello dataset rendering
- **License**: Apache-2.0
- **Reason for bundling**: Not available on PyPI, patched for numpy 2.0 compatibility

### Dual Directory Structure

LayerD maintains vendored dependencies in two locations:

- **`vendor/`**: Source of truth for git subtree operations (tracked in git)
- **`src/layerd/_vendor/`**: Bundled copy for distribution (tracked in git)

Both directories are committed to git to ensure `pip install git+...` and editable installs work correctly.

The `_vendor` prefix indicates these are internal dependencies and should not be imported directly by users.

See [development.md#vendored-dependencies](development.md#vendored-dependencies) for syncing instructions.

## Type System

LayerD uses strict mypy type checking:

- `disallow_untyped_defs=true`
- `disallow_incomplete_defs=true`
- `no_implicit_optional=true`

All functions must have complete type annotations. This helps catch bugs early and provides better IDE support.

## Related Documentation

- [Development Guide](development.md) - Development setup and workflows
- [Training Guide](training.md) - Training the matting module
- [Evaluation Guide](evaluation.md) - Evaluation metrics and usage
