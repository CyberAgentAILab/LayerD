# Installation Guide

This guide covers all installation options for LayerD, from basic inference to full development setup.

## Requirements

### System Requirements

We have verified reproducibility under the following environment:

- **OS**: Ubuntu 22.04 (should work on other Linux distributions and macOS)
- **Python**: 3.12.3 or compatible
- **CUDA**: 12.8 (optional, for GPU acceleration)
- **Package Manager**: [uv](https://docs.astral.sh/uv/) 0.8.17 (for development)

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only inference is supported
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM for faster inference
- **Training**: NVIDIA GPU with 16GB+ VRAM (A100 40GB recommended for efficient training)

## Installation Options

### Option 1: Inference Only (Recommended for Most Users)

Install the core package with inference and evaluation capabilities:

```bash
pip install git+https://github.com/CyberAgentAILab/LayerD.git
```

This is suitable if you only want to:

- Decompose images into layers
- Run evaluations on decomposed layers
- Use the CLI or Python API

### Option 2: With Dataset Generation

If you want to generate training datasets from Crello:

```bash
pip install "git+https://github.com/CyberAgentAILab/LayerD.git#egg=layerd[dataset]"
```

This includes:

- All inference capabilities
- Crello dataset utilities
- Dataset generation tools

### Option 3: With Training

If you want to train or fine-tune models:

```bash
pip install "git+https://github.com/CyberAgentAILab/LayerD.git#egg=layerd[train]"
```

This includes:

- All inference capabilities
- Training framework and utilities
- PyTorch distributed training support

### Option 4: All Features

Install with all optional dependencies:

```bash
pip install "git+https://github.com/CyberAgentAILab/LayerD.git#egg=layerd[all]"
```

This includes everything: inference, dataset generation, and training.

### Option 5: Development Setup (For Contributors)

LayerD uses [uv](https://docs.astral.sh/uv/) to manage the development environment.

1. **Install uv** (if not already installed):

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

1. **Clone and install**:

```bash
git clone https://github.com/CyberAgentAILab/LayerD.git
cd LayerD
uv sync --all-extras
```

This installs all dependencies including development tools (pytest, mypy, ruff). Dev tools are included by default with `uv sync`.

See [development.md](development.md) for more details on development workflows.

## Model Downloads

On first run, LayerD automatically downloads pre-trained models:

1. **BiRefNet matting model** (~1GB)
   - Source: HuggingFace repository [cyberagent/layerd-birefnet](https://huggingface.co/cyberagent/layerd-birefnet)
   - Cached in: `~/.cache/huggingface/hub/`

2. **LaMa inpainting model** (~200MB)
   - Source: GitHub repository [enesmsahin/simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting)
   - Cached in: `~/.cache/simple_lama_inpainting/`

Please ensure you have:

- Stable internet connection during first run
- At least 2GB free disk space for models
- Access to HuggingFace and GitHub (some networks may block these)

### Offline Usage

To use LayerD offline after initial setup:

1. Run LayerD once with internet to download models
2. Models will be cached locally
3. Subsequent runs work offline

To manually download models:

```python
from layerd import LayerD

# This downloads and caches models
layerd = LayerD(matting_hf_card="cyberagent/layerd-birefnet").to("cpu")
```

## Verification

Verify your installation by running a simple test:

```python
from PIL import Image
from layerd import LayerD

# Create a LayerD instance (downloads models if needed)
layerd = LayerD(matting_hf_card="cyberagent/layerd-birefnet").to("cpu")
print("LayerD installed successfully!")
```

Or use the CLI:

```bash
layerd --help
```

You should see the CLI help message without errors.

## Troubleshooting

If you encounter issues during installation, see the [Troubleshooting Guide](troubleshooting.md) for detailed solutions:

- **Dependency conflicts**: Use a fresh virtual environment
- **uv not found**: Install uv first
- **Model download fails**: Check internet connection, try manual download, or configure proxy
- **No space left**: Free up 2GB+ or change cache directory
- **CUDA issues**: Use CPU mode or reduce process size

For complete troubleshooting steps, see [troubleshooting.md](troubleshooting.md).

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade git+https://github.com/CyberAgentAILab/LayerD.git
```

For development installations:

```bash
cd LayerD
git pull
uv sync --all-extras --all-groups
```

## Next Steps

- [Quick start with inference](../README.md#quick-example)
- [Inference guide](inference.md)
- [Training guide](training.md)
- [Development guide](development.md)
