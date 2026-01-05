# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LayerD is a layer decomposition method that extracts editable layers from raster graphic design images. The system uses a two-stage iterative approach:

1. **Top-layer matting**: Extracts the alpha matte of the topmost layer using BiRefNet
2. **Inpainting**: Fills in the removed content using LaMa to reconstruct the background

The main `LayerD` class orchestrates this pipeline iteratively to decompose an image into multiple layers (background + foreground layers).

## Development Commands

### Environment Setup

```bash
# Install dependencies (uses uv for package management)
uv sync

# Install with dev dependencies
uv sync --extra dev
```

### Testing

```bash
# Run all tests
uv run pytest

# Run tests with image output saved
uv run pytest --save-images

# Run tests with custom matting process size
uv run pytest --matting-process-size 512 512

# Run specific test
uv run pytest tests/test_basic_decompose.py::test_decompose
```

### Code Quality

```bash
# Run type checking
uv run mypy src/

# Run linting
uv run ruff check src/

# Format code
uv run ruff format src/
```

### Inference

```bash
# Basic inference
uv run python ./tools/infer.py \
  --input <path/to/image> \
  --output-dir <output/path> \
  --device cpu

# Batch inference with glob pattern
uv run python ./tools/infer.py \
  --input "data/*.png" \
  --output-dir outputs/ \
  --device cuda
```

### Training

#### Dataset Preparation

```bash
# Generate Crello matting dataset
uv run python ./tools/generate_crello_matting.py \
  --output-dir <dataset/path> \
  --inpainting \
  --save-layers
```

#### Training Commands

```bash
# Single GPU training
uv run python ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=<dataset/path> \
  out_dir=<output/path> \
  device=cuda

# Multi-GPU training with torch.distributed
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node 2 \
  ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=<dataset/path> \
  out_dir=<output/path> \
  dist=true

# Multi-GPU training with Hugging Face Accelerate
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node 2 \
  ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=<dataset/path> \
  out_dir=<output/path> \
  use_accelerate=true \
  mixed_precision=bf16
```

Training configuration uses Hydra and can be overridden via command-line arguments. `data_root` and `out_dir` are mandatory runtime parameters.

### Evaluation

```bash
# Run evaluation on dataset
uv run python ./tools/evaluate.py \
  --pred-dir <predictions/path> \
  --gt-dir <groundtruth/path> \
  --output-dir <results/path> \
  --max-edits 5
```

## Code Architecture

### Core Components

**Main Pipeline ([src/layerd/models/layerd.py](src/layerd/models/layerd.py))**

- `LayerD` class: Main interface for layer decomposition
- `decompose()`: Iteratively extracts layers (max 3 iterations by default)
- `_decompose_step()`: Single iteration of matting + inpainting
- Uses helper functions from `helpers.py` for refinement operations

**Model Abstraction**

- Base classes define interfaces: `BaseMatting` and `BaseInpaint`
- Registry pattern in `models/matting/__init__.py` and `models/inpaint/__init__.py`
- Use `build_matting()` and `build_inpaint()` factory functions to instantiate models
- Currently supports: BiRefNet for matting, LaMa for inpainting

**Refinement Pipeline**

The decomposition includes optional refinement steps controlled by flags:

- `use_unblend`: Estimates foreground color by unblending (subtracting background)
- `fg_refine`: Refines foreground alpha and colors using flat color region detection
- `bg_refine`: Refines background with palette-based color assignment

**Evaluation ([src/layerd/evaluation/](src/layerd/evaluation/))**

- `LayersEditDist`: Main metric for layer decomposition quality
- Uses Dynamic Time Warping (DTW) to align predicted and ground truth layers
- Computes edit distance between layer sequences

### Module Organization

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
└── configs/               # Hydra configuration files
    └── train.yaml         # Training hyperparameters
```

### Key Design Patterns

1. **Factory Pattern**: Models are created via `build_matting()` and `build_inpaint()` functions with string identifiers
2. **Abstract Base Classes**: All models inherit from `BaseMatting` or `BaseInpaint` with validation
3. **Iterative Decomposition**: `decompose()` runs `_decompose_step()` until no more layers or max iterations reached
4. **PIL Image Interface**: Main API uses PIL Images; internal processing uses numpy arrays

### Important Implementation Details

- **Input Requirements**: Prefer PNG images to avoid compression artifacts around text edges
- **Model Downloads**: First run downloads models from HuggingFace (BiRefNet ~1GB) and GitHub (LaMa ~200MB)
- **Alpha Format**: Matting models output float64 alpha in [0, 1] range
- **Mask Expansion**: Uses `kernel_scale` parameter (default 0.015) to expand masks based on image dimensions
- **Layer Order**: `decompose()` returns [background, topmost_fg, ..., bottommost_fg]

## Type Checking

This codebase uses strict mypy configuration:

- `disallow_untyped_defs=true`
- `disallow_incomplete_defs=true`
- `no_implicit_optional=true`

All functions must have complete type annotations.

## Testing Notes

- Test configuration uses pytest fixtures in `tests/conftest.py`
- Custom options: `--save-images` and `--matting-process-size`
- Test outputs saved to `tests/output/` (gitignored)
- Pytest filters FutureWarnings from timm library (configured in pyproject.toml)
