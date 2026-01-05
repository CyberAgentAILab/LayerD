# Troubleshooting Guide

Common issues and solutions when using LayerD.

## Installation Issues

### Dependency conflicts

```
ERROR: Cannot install layerd because these package versions have conflicting dependencies.
```

**Solutions:**

```bash
# Use fresh virtual environment
python -m venv layerd-env
source layerd-env/bin/activate  # Windows: layerd-env\Scripts\activate
pip install git+https://github.com/CyberAgentAILab/LayerD.git

# Or upgrade pip first
pip install --upgrade pip setuptools wheel
pip install git+https://github.com/CyberAgentAILab/LayerD.git

# With conda
conda create -n layerd python=3.12
conda activate layerd
pip install git+https://github.com/CyberAgentAILab/LayerD.git
```

### uv not found

```bash
bash: uv: command not found
```

**Solution:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or
pip install uv
```

### NumPy version conflicts

**Solution:** LayerD requires numpy 2.0+

```bash
pip install --upgrade "numpy>=2.0"
```

## Model Download Issues

### Download fails or times out

**Solutions:**

```bash
# Check internet connection, then try manual download
python -c "from huggingface_hub import snapshot_download; snapshot_download('cyberagent/layerd-birefnet')"

# If behind proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# Use HuggingFace CLI for better resume support
pip install huggingface_hub[cli]
huggingface-cli download cyberagent/layerd-birefnet
```

### No space left on device

**Solution:** Free up 2GB+ or change cache directory

```bash
export HF_HOME=/path/to/new/cache
export TORCH_HOME=/path/to/new/cache
```

## CUDA and GPU Issues

### CUDA out of memory

**Solutions:**

```python
# Reduce process size
layerd = LayerD(matting_process_size=(512, 512))

# Or use CPU
layerd = layerd.to("cpu")

# For training: reduce batch size or use mixed precision
```

```bash
uv run python ./tools/train.py ... batch_size=2 mixed_precision=bf16
```

### CUDA not available

**Solutions:**

```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Reinstall PyTorch with CUDA (see https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Check NVIDIA driver
nvidia-smi
```

### cuDNN errors

**Solutions:**

1. Update GPU drivers
2. Reinstall PyTorch: `pip uninstall torch torchvision && pip install torch torchvision`
3. Check hardware: `nvidia-smi`
4. Try CPU mode to isolate issue: `layerd.to("cpu")`

## Inference Issues

### Poor quality around text edges

**Solutions:**

```python
# Use PNG input (not JPEG) to avoid compression artifacts
image = Image.open("design.png")

# Increase kernel_scale for better edge handling
layerd = LayerD(kernel_scale=0.020)  # Default: 0.015

# Increase matting process size
layerd = LayerD(matting_process_size=(1024, 1024))
```

### Too few or too many layers

**Solution:** Adjust max_iterations

```python
layers = layerd.decompose(image, max_iterations=5)  # More layers
layers = layerd.decompose(image, max_iterations=2)  # Fewer layers
```

### Inference is very slow

**Solutions:**

```python
# Use GPU
layerd = layerd.to("cuda")

# Reduce process size
layerd = LayerD(matting_process_size=(512, 512))

# Reduce iterations
layers = layerd.decompose(image, max_iterations=2)
```

### Cannot identify image file

**Solutions:**

```bash
# Verify file is valid
file your_image.png

# Try with OpenCV
python -c "import cv2; img = cv2.imread('image.png'); print(img.shape)"

# Check permissions and re-download if needed
```

## Training Issues

### Training loss is NaN

**Solutions:**

```bash
# Reduce learning rate
uv run python ./tools/train.py ... learning_rate=5e-5

# Use mixed precision with bf16
uv run python ./tools/train.py ... mixed_precision=bf16

# Check dataset for corrupted images
```

### Training is very slow

**Solutions:**

```bash
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node 2 \
  ./tools/train.py ... dist=true

# Use mixed precision
uv run python ./tools/train.py ... mixed_precision=bf16

# Increase data loading workers
uv run python ./tools/train.py ... num_workers=8
```

### Dataset preparation fails

**Solutions:**

1. Check internet connection (downloads ~20GB)
2. Ensure sufficient disk space (~100GB)
3. Verify HuggingFace access: `python -c "from datasets import load_dataset; load_dataset('cyberagent/crello')"`

### Multi-GPU not using all GPUs

**Solutions:**

```bash
# Verify dist=true is set
uv run torchrun ... dist=true

# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node 4 ...

# Verify nproc_per_node matches GPU count
```

## Evaluation Issues

### Mismatched layer counts warning

This is expected. `LayersEditDist` handles different layer counts automatically.

### Evaluation is very slow

**Solutions:**

```python
# Evaluate subset first
samples = list(pred_dir.iterdir())[:100]

# Use multiprocessing (modify evaluation script)
# Use smaller images if pixel-perfect accuracy not needed
```

### High edit distance despite good visuals

**Solutions:**

1. Check layer ordering (background should be first)
2. Verify alpha quality: `compute_alpha_iou(layer_pred, layer_gt)` should be > 0.8
3. Check for extra/missing layers

## General Issues

### Import errors

```python
ModuleNotFoundError: No module named 'layerd'
```

**Solutions:**

```bash
# Verify installation
pip list | grep layerd

# Reinstall
pip uninstall layerd && pip install git+https://github.com/CyberAgentAILab/LayerD.git

# Check environment
which python

# For development
cd LayerD && uv sync --all-extras
```

### Type checking errors

LayerD requires strict type annotations:

```python
# Bad
def process(image):
    ...

# Good
def process(image: Image.Image) -> list[Image.Image]:
    ...
```

See [development.md](development.md) for details.

### Permission denied

**Solutions:**

```bash
# Check permissions
ls -la /path/to/file

# Make writable
chmod +w /path/to/output

# Don't run as root
```

## Getting Help

If you can't resolve your issue:

1. Check [GitHub issues](https://github.com/CyberAgentAILab/LayerD/issues)
2. Create new issue with: LayerD version, Python version, OS, full traceback, minimal example
3. Read the [paper](https://arxiv.org/abs/2509.25134) for methodology details
4. Check related docs: [Installation](installation.md), [Inference](inference.md), [Training](training.md), [Development](development.md)
