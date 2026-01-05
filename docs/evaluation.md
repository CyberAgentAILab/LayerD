# Evaluation Guide

This guide covers how to evaluate layer decomposition quality using LayerD's evaluation metrics.

## Overview

LayerD provides a comprehensive evaluation framework based on:

- **LayersEditDist**: Sequence-based edit distance metric using Dynamic Time Warping (DTW)
- **Per-layer metrics**: RGBL1 (color accuracy) and AlphaIoU (mask accuracy)

These metrics compare predicted layer decompositions against ground-truth layers.

## Quick Start

### Python API

Evaluate layer decomposition quality:

```python
from layerd.evaluation import LayersEditDist
from PIL import Image

# Load layers (both should be lists of PIL Images in RGBA format)
layers_pred = [Image.open(f"pred/layer_{i}.png") for i in range(3)]
layers_gt = [Image.open(f"gt/layer_{i}.png") for i in range(3)]

# Create metric
metric = LayersEditDist()

# Compute edit distance
result = metric(layers_pred, layers_gt)

print(f"Edit Distance: {result['edit_distance']}")
print(f"Normalized Edit Distance: {result['normalized_edit_distance']}")
```

### Command-Line Evaluation

For dataset-level evaluation:

```bash
uv run python ./tools/evaluate.py \
  --pred-dir /path/to/predictions \
  --gt-dir /path/to/groundtruth \
  --output-dir /path/to/results \
  --max-edits 5
```

## LayersEditDist Metric

### What It Measures

The `LayersEditDist` metric computes the minimum edit distance between two layer sequences:

- **Insert**: Adding a missing layer
- **Delete**: Removing an extra layer
- **Modify**: Changing an existing layer

The metric uses Dynamic Time Warping (DTW) to align layers and compute the optimal sequence of edit operations.

### Why It's Useful

Traditional per-layer metrics (like MSE or IoU) require exact correspondence between predicted and ground-truth layers. `LayersEditDist` handles:

- Different numbers of layers
- Misaligned layer orders
- Missing or extra layers

This makes it suitable for evaluating layer decomposition where the number and order of layers may vary.

### Output Format

```python
result = metric(layers_pred, layers_gt)
# result is a dictionary:
{
    'edit_distance': 2.5,           # Raw edit distance
    'normalized_edit_distance': 0.5, # Normalized by max(len(pred), len(gt))
    'alignment': [...],              # DTW alignment path
    'operation_costs': [...]         # Cost of each edit operation
}
```

### Interpretation

- **Edit distance = 0**: Perfect match (identical layers)
- **Lower is better**: Smaller edit distance means better quality
- **Normalized range**: [0, max_edits] where max_edits is typically 3-5

## Per-Layer Metrics

### RGBL1 - Color Accuracy

Measures the L1 distance between RGB values of predicted and ground-truth layers:

```python
from layerd.evaluation.metrics import compute_rgbl1

rgbl1 = compute_rgbl1(layer_pred, layer_gt)
# Returns float in [0, 1], lower is better
```

### AlphaIoU - Mask Accuracy

Measures the Intersection over Union (IoU) of alpha channels:

```python
from layerd.evaluation.metrics import compute_alpha_iou

alpha_iou = compute_alpha_iou(layer_pred, layer_gt)
# Returns float in [0, 1], higher is better
```

### Combined Per-Layer Evaluation

```python
from layerd.evaluation.metrics import evaluate_layer_pair

metrics = evaluate_layer_pair(layer_pred, layer_gt)
print(f"RGBL1: {metrics['rgbl1']:.4f}")
print(f"AlphaIoU: {metrics['alpha_iou']:.4f}")
```

## Dataset-Level Evaluation

### Directory Structure

The evaluation script expects the following structure:

```
predictions/
├── sample_001/
│   ├── 0000.png  # Background
│   ├── 0001.png  # Layer 1
│   ├── 0002.png  # Layer 2
│   └── ...
├── sample_002/
│   ├── 0000.png
│   └── ...
└── ...

groundtruth/
├── sample_001/
│   ├── 0000.png
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── ...
```

Each sample has its own directory with numbered layer files (0000.png, 0001.png, etc.).

### Running Evaluation

```bash
uv run python ./tools/evaluate.py \
  --pred-dir /path/to/predictions \
  --gt-dir /path/to/groundtruth \
  --output-dir /path/to/results \
  --max-edits 5
```

### Evaluation Script Options

- **`--pred-dir`** (required): Directory with predicted layers
- **`--gt-dir`** (required): Directory with ground-truth layers
- **`--output-dir`** (required): Directory to save evaluation results
- **`--max-edits`**: Maximum edit distance for normalization (default: 5)

### Output Files

The script saves results to `output-dir/`:

```
results/
├── summary.json          # Overall statistics
├── per_sample.csv        # Per-sample metrics
└── alignment_viz/        # Visualization of layer alignments (optional)
```

**summary.json** contains:

```json
{
  "mean_edit_distance": 1.23,
  "mean_normalized_edit_distance": 0.41,
  "mean_rgbl1": 0.15,
  "mean_alpha_iou": 0.87,
  "num_samples": 100
}
```

## Crello Dataset Evaluation

The Crello dataset prepared by [generate_crello_matting.py](../tools/generate_crello_matting.py) includes a `layers/` directory ready for evaluation.

### Preparing Crello for Evaluation

When generating the dataset, use `--save-layers`:

```bash
uv run python ./tools/generate_crello_matting.py \
  --output-dir /path/to/dataset \
  --inpainting \
  --save-layers
```

This creates:

```
dataset/
├── train/
│   └── layers/  # Ground-truth layers for training set
├── validation/
│   └── layers/
└── test/
    └── layers/  # Ground-truth layers for test set
```

### Evaluating on Crello

```bash
# 1. Run inference on test set
uv run python ./tools/infer.py \
  --input /path/to/dataset/test/composite/ \
  --output-dir /path/to/predictions/ \
  --device cuda

# 2. Evaluate predictions
uv run python ./tools/evaluate.py \
  --pred-dir /path/to/predictions/ \
  --gt-dir /path/to/dataset/test/layers/ \
  --output-dir /path/to/eval_results/ \
  --max-edits 5
```

## Advanced Usage

### Custom Evaluation Metrics

Add custom metrics by extending the evaluation framework:

```python
from layerd.evaluation.metrics import evaluate_layer_pair
from PIL import Image
import numpy as np

def custom_metric(layer_pred, layer_gt):
    """Custom evaluation metric"""
    # Convert PIL to numpy
    pred_arr = np.array(layer_pred)
    gt_arr = np.array(layer_gt)

    # Your metric computation
    score = ...

    return score

# Use in evaluation
layer_pred = Image.open("pred.png")
layer_gt = Image.open("gt.png")
score = custom_metric(layer_pred, layer_gt)
```

### Visualizing Alignments

Visualize DTW alignment between predicted and ground-truth layers:

```python
from layerd.evaluation import LayersEditDist
import matplotlib.pyplot as plt

metric = LayersEditDist()
result = metric(layers_pred, layers_gt)

# Get alignment path
alignment = result['alignment']

# Visualize (example)
for i, (pred_idx, gt_idx) in enumerate(alignment):
    print(f"Step {i}: Pred layer {pred_idx} <-> GT layer {gt_idx}")
```

### Batch Evaluation Script

For large-scale evaluation:

```python
from pathlib import Path
from layerd.evaluation import LayersEditDist
from PIL import Image

pred_root = Path("/path/to/predictions")
gt_root = Path("/path/to/groundtruth")

metric = LayersEditDist()
results = []

for sample_dir in pred_root.iterdir():
    if not sample_dir.is_dir():
        continue

    sample_id = sample_dir.name

    # Load predicted layers
    layers_pred = []
    for layer_file in sorted(sample_dir.glob("*.png")):
        layers_pred.append(Image.open(layer_file))

    # Load ground-truth layers
    gt_dir = gt_root / sample_id
    layers_gt = []
    for layer_file in sorted(gt_dir.glob("*.png")):
        layers_gt.append(Image.open(layer_file))

    # Evaluate
    result = metric(layers_pred, layers_gt)
    results.append({
        'sample_id': sample_id,
        'edit_distance': result['edit_distance'],
        'normalized_edit_distance': result['normalized_edit_distance']
    })

# Compute statistics
mean_edit_dist = sum(r['edit_distance'] for r in results) / len(results)
print(f"Mean Edit Distance: {mean_edit_dist:.4f}")
```

## Troubleshooting

### Common Issues

**Problem**: "Mismatched layer counts"

**Solution**: This is expected behavior. `LayersEditDist` handles different layer counts automatically. The edit distance reflects the cost of aligning mismatched sequences.

**Problem**: Evaluation is slow for large datasets

**Solution**:

1. Use multiprocessing to parallelize sample evaluation
2. Evaluate on a subset first to verify correctness
3. Use smaller image resolutions if per-pixel accuracy is not critical

**Problem**: High edit distance despite visually similar results

**Solution**:

- Check layer ordering (background should be first)
- Verify alpha channel quality (check AlphaIoU)
- Adjust edit operation costs if needed (modify metric parameters)

For more troubleshooting help, see [troubleshooting.md](troubleshooting.md).

## Metric Details

### Edit Operation Costs

The `LayersEditDist` metric uses the following default costs:

- **Insert**: 1.0 (cost of adding a missing layer)
- **Delete**: 1.0 (cost of removing an extra layer)
- **Modify**: Based on RGBL1 + (1 - AlphaIoU)

These can be customized by modifying the metric implementation.

### Dynamic Time Warping (DTW)

DTW finds the optimal alignment between two sequences by:

1. Computing pairwise distance matrix between all layer pairs
2. Finding the minimum-cost path through the matrix
3. Returning the alignment and total cost

See [src/layerd/evaluation/dtw.py](../src/layerd/evaluation/dtw.py) for implementation details.

## Related Documentation

- [Inference Guide](inference.md) - Generate predictions to evaluate
- [Training Guide](training.md) - Train models for evaluation
- [Architecture](architecture.md) - Understanding evaluation components
- [Paper](https://arxiv.org/abs/2509.25134) - Full metric description
