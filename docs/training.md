# Training Guide

This guide covers how to train and fine-tune the top-layer matting module of LayerD using the Crello dataset.

## Overview

LayerD's matting module is based on [BiRefNet](https://github.com/ZhengPeng7/BiRefNet), which can be fine-tuned on custom datasets. We provide:

- Dataset preparation tools for the Crello dataset
- Training scripts with Hydra configuration
- Support for single-GPU and multi-GPU training
- Integration with torch.distributed and Hugging Face Accelerate

## Prerequisites

### Installation

Install LayerD with training dependencies:

```bash
pip install "git+https://github.com/CyberAgentAILab/LayerD.git#egg=layerd[train]"
```

Or for development:

```bash
git clone https://github.com/CyberAgentAILab/LayerD.git
cd LayerD
uv sync --all-extras --all-groups
```

### Hardware Requirements

- **Minimum**: NVIDIA GPU with 16GB VRAM
- **Recommended**: NVIDIA A100 40GB or similar
- **Multi-GPU**: 2-4 GPUs for faster training

### Dataset Requirements

- **Storage**: ~20GB for Crello dataset
- **Additional**: ~50-100GB for generated training dataset
- **Internet**: Stable connection for initial dataset download

## Dataset Preparation

### Downloading and Preparing Crello Dataset

The Crello dataset is available on [HuggingFace](https://huggingface.co/datasets/cyberagent/crello) and can be automatically downloaded and converted:

```bash
uv run python ./tools/generate_crello_matting.py \
  --output-dir /path/to/dataset \
  --inpainting \
  --save-layers
```

### Script Options

- **`--output-dir`** (required): Output directory for prepared dataset
- **`--inpainting`**: Apply inpainting to create intermediate composite images
- **`--save-layers`**: Save individual layers (useful for evaluation)

### Dataset Structure

After preparation, the dataset has the following structure:

```
/path/to/dataset/
├── train/
│   ├── im/          # Input images (full composite or intermediate composite)
│   ├── gt/          # Ground-truth alpha mattes (top-layer)
│   ├── composite/   # Full composite images (for evaluation)
│   └── layers/      # Ground-truth RGBA layers (for evaluation)
├── validation/
│   ├── im/
│   ├── gt/
│   ├── composite/
│   └── layers/
└── test/
    ├── im/
    ├── gt/
    ├── composite/
    └── layers/
```

**Directories explained:**

- **`im/`**: Input images for training (what the model sees)
- **`gt/`**: Ground-truth alpha mattes (what the model predicts)
- **`composite/`**: Full composite images (not used in training, for evaluation)
- **`layers/`**: Individual layers (not used in training, for evaluation)

### Dataset Generation Time

First run downloads the Crello dataset (~20GB) from HuggingFace:

- Download time: Depends on connection speed (30 min - 2 hours)
- Processing time: ~2-4 hours on a modern CPU

## Training Configuration

LayerD uses [Hydra](https://hydra.cc/) for configuration management. The default configuration is in [src/layerd/configs/train.yaml](../src/layerd/configs/train.yaml).

### Configuration File Structure

Key configuration parameters:

```yaml
# Data
data_root: ???  # Must be specified at runtime
batch_size: 4
num_workers: 4

# Model
model_name: "birefnet"
backbone: "pvt_v2_b2"

# Training
epochs: 100
learning_rate: 1e-4
weight_decay: 1e-4

# Optimization
optimizer: "adamw"
scheduler: "cosine"

# Distributed training
dist: false
use_accelerate: false
mixed_precision: "no"

# Output
out_dir: ???  # Must be specified at runtime
save_interval: 10
```

### Required Parameters

These parameters must be specified at runtime:

- **`data_root`**: Path to prepared dataset directory
- **`out_dir`**: Path to save training outputs (checkpoints, logs)

## Training

### Single GPU Training

Basic training on a single GPU:

```bash
uv run python ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=/path/to/dataset \
  out_dir=/path/to/output \
  device=cuda
```

### Multi-GPU Training with torch.distributed

For distributed training across multiple GPUs using `torch.distributed`:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node 2 \
  ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=/path/to/dataset \
  out_dir=/path/to/output \
  dist=true
```

**Parameters:**

- `CUDA_VISIBLE_DEVICES`: Specify which GPUs to use
- `--nproc_per_node`: Number of processes (should match number of GPUs)
- `dist=true`: Enable distributed training mode

### Multi-GPU Training with Hugging Face Accelerate

For distributed training with mixed precision using Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node 2 \
  ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=/path/to/dataset \
  out_dir=/path/to/output \
  use_accelerate=true \
  mixed_precision=bf16
```

**Mixed precision options:**

- `no`: Full precision (float32)
- `fp16`: Half precision (float16)
- `bf16`: BFloat16 (recommended for A100 GPUs)

### Overriding Configuration

Override any configuration parameter from command line:

```bash
uv run python ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=/path/to/dataset \
  out_dir=/path/to/output \
  batch_size=8 \
  learning_rate=5e-5 \
  epochs=50 \
  device=cuda
```

### Example Training Commands

**Quick test (single GPU, small dataset):**

```bash
uv run python ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=/path/to/dataset \
  out_dir=./outputs/test_run \
  epochs=10 \
  device=cuda
```

**Production training (4 GPUs, full dataset, mixed precision):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node 4 \
  ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=/path/to/dataset \
  out_dir=./outputs/production \
  use_accelerate=true \
  mixed_precision=bf16 \
  batch_size=4 \
  epochs=100
```

## Training Time and Resources

### Expected Training Time

With default configuration ([train.yaml](../src/layerd/configs/train.yaml)):

- **4x A100 40GB** with `mixed_precision=bf16`: ~40 hours
- **2x A100 40GB** with `mixed_precision=bf16`: ~80 hours
- **1x A100 40GB** with `mixed_precision=bf16`: ~160 hours

Training time scales approximately linearly with number of GPUs.

### Resource Usage

- **GPU Memory**:
  - Single GPU (bf16): ~35GB
  - Single GPU (fp32): ~40GB+ (may not fit in smaller GPUs)
- **Disk Space**:
  - Checkpoints: ~1GB per checkpoint
  - Logs and metrics: ~100MB
  - Total: ~10-15GB for full training run

## Monitoring Training

### Training Outputs

Training saves outputs to the specified `out_dir`:

```
out_dir/
├── checkpoints/
│   ├── epoch_10.pth
│   ├── epoch_20.pth
│   └── best.pth
├── logs/
│   └── train.log
└── config.yaml  # Saved configuration
```

### Logging

Training logs include:

- Loss values (per batch and per epoch)
- Learning rate schedule
- Training/validation metrics
- Timing information

View logs in real-time:

```bash
tail -f out_dir/logs/train.log
```

### TensorBoard (Optional)

If you want to use TensorBoard for visualization, modify the training script to add TensorBoard logging.

## Using Trained Models

### Loading Custom Weights for Inference

After training, use your custom weights for inference:

```bash
uv run python ./tools/infer.py \
  --input "data/*.png" \
  --output-dir outputs/ \
  --matting-weight-path /path/to/output/checkpoints/best.pth \
  --device cuda
```

Or in Python:

```python
from layerd import LayerD

# Load with custom weights
layerd = LayerD(
    matting_hf_card="cyberagent/layerd-birefnet",  # Base model architecture
    # Then manually load your weights after initialization
)
# Note: Direct custom weight loading API may need implementation
```

### Evaluating Trained Models

Evaluate your trained model on test set:

```bash
# First, run inference on test set
uv run python ./tools/infer.py \
  --input /path/to/dataset/test/composite/ \
  --output-dir /path/to/predictions/ \
  --matting-weight-path /path/to/output/checkpoints/best.pth \
  --device cuda

# Then evaluate predictions
uv run python ./tools/evaluate.py \
  --pred-dir /path/to/predictions/ \
  --gt-dir /path/to/dataset/test/layers/ \
  --output-dir /path/to/eval_results/ \
  --max-edits 5
```

See [evaluation.md](evaluation.md) for more details.

## Troubleshooting

### Common Issues

**Problem**: CUDA out of memory during training

**Solutions**:

1. Reduce batch size: `batch_size=2` or `batch_size=1`
2. Use mixed precision: `mixed_precision=bf16`
3. Use gradient accumulation (modify training script)

**Problem**: Training is very slow

**Solutions**:

1. Use multiple GPUs with `dist=true` or `use_accelerate=true`
2. Use mixed precision: `mixed_precision=bf16`
3. Increase `num_workers` for data loading: `num_workers=8`

**Problem**: NaN loss during training

**Solutions**:

1. Reduce learning rate: `learning_rate=5e-5`
2. Check dataset for corrupted images
3. Use mixed precision with bf16 instead of fp16

**Problem**: Dataset download fails

**Solutions**:

1. Check internet connection
2. Ensure sufficient disk space (~20GB)
3. Try again (downloads can be resumed)

For more troubleshooting help, see [troubleshooting.md](troubleshooting.md).

## Fine-tuning Tips

### Starting from Pre-trained Weights

The default configuration uses pre-trained BiRefNet weights. Fine-tuning from these weights is recommended for:

- Faster convergence
- Better performance on small datasets
- Domain-specific customization

### Hyperparameter Tuning

Key hyperparameters to adjust:

- **Learning rate**: Start with `1e-4`, reduce if training is unstable
- **Batch size**: Larger is better (if memory allows)
- **Epochs**: Monitor validation loss to determine optimal stopping point
- **Weight decay**: Controls regularization (default: `1e-4`)

### Data Augmentation

The training pipeline includes data augmentation. You can modify augmentation in the dataset implementation:

- Horizontal flipping
- Random cropping
- Color jittering
- Scaling

See [src/layerd/matting/birefnet/dataset.py](../src/layerd/matting/birefnet/dataset.py) for details.

## Advanced Topics

### Custom Dataset

To train on a custom dataset:

1. Prepare your dataset in the same structure as Crello:

   ```
   dataset/
   ├── train/
   │   ├── im/   # Input composite images
   │   └── gt/   # Ground-truth alpha mattes
   └── validation/
       ├── im/
       └── gt/
   ```

2. Ensure images are paired (same filename in `im/` and `gt/`)

3. Run training with your dataset path:

   ```bash
   uv run python ./tools/train.py \
     config_path=./src/layerd/configs/train.yaml \
     data_root=/path/to/custom_dataset \
     out_dir=/path/to/output \
     device=cuda
   ```

### Distributed Training on Multiple Nodes

For multi-node distributed training, modify the `torchrun` command:

```bash
# On each node, run:
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  ./tools/train.py \
  config_path=./src/layerd/configs/train.yaml \
  data_root=/path/to/dataset \
  out_dir=/path/to/output \
  dist=true
```

## Acknowledgments

We thank the authors of [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) for releasing their code, which we used as the basis for our matting backbone.

## Related Documentation

- [Installation Guide](installation.md) - Setup for training
- [Evaluation Guide](evaluation.md) - Evaluate trained models
- [Architecture](architecture.md) - Understanding the model
- [Development Guide](development.md) - Development workflows
