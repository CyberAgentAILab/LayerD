<div align="center">
<h1> LayerD: Decomposing Raster Graphic Designs into Layers </h1>

<h4 align="center">
    <a href="https://tomoyukun.github.io/biography/">Tomoyuki Suzuki</a><sup>1</sup>&emsp;
    <a href="">Kang-Jun Liu</a><sup>2</sup>&emsp;
    <a href="https://naoto0804.github.io/">Naoto Inoue</a><sup>1</sup>&emsp;
    <a href="https://sites.google.com/view/kyamagu">Kota Yamaguchi</a><sup>1</sup>&emsp;
    <br>
    <br>
    <sup>1</sup>CyberAgent, <sup>2</sup>Tohoku University
</h4>

<h2 align="center">
ICCV 2025
</h2>

</div>

<div align="center">

[![arxiv paper](https://img.shields.io/badge/arxiv-paper-orange)](https://arxiv.org/abs/2509.25134)
<a href='https://cyberagentailab.github.io/LayerD/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

</div>

![LayerD layer decomposition example](static/teaser.png)

LayerD is a layer decomposition method that extracts editable layers from raster graphic design images. This repository contains the official implementation of our ICCV 2025 paper.

See our [project page](https://cyberagentailab.github.io/LayerD/) for more details.

## Installation

Install LayerD with pip:

```bash
pip install git+https://github.com/CyberAgentAILab/LayerD.git
```

For other installation options (dataset generation, training, development), see the [Installation Guide](docs/installation.md).

## Quick Start

Decompose an image into layers and export to SVG:

```python
from layerd import LayerDPipeline
from PIL import Image

pipeline = LayerDPipeline(device="cpu")
image = Image.open("./data/test_image_2.png")
result = pipeline(image)
result.save("output.svg")
```

The pipeline handles decomposition, organization, and export in one call. Results include organized elements with type classification (text/vector/image).

## Features

- **Layer Decomposition** - BiRefNet-based matting extracts clean layers from raster designs
- **SVG Export** - Generate scalable vector graphics with embedded or external images
- **PSD Export** - Create Photoshop documents with editable layers
- **Element Classification** - Automatic detection of text, vector, and image elements
- **Unified Pipeline API** - Complete workflow from image to export in 3 lines of code
- **Custom Weights** - Load fine-tuned models from local or remote storage
- **OCR Support** - Text detection and recognition (coming soon)

## Advanced Usage

For fine-grained control, use the low-level API:

```python
from layerd import LayerD

layerd = LayerD(matting_hf_card="cyberagent/layerd-birefnet").to("cpu")
layers = layerd.decompose(image)
# ... custom postprocessing ...
```

See the [Inference Guide](docs/inference.md) for low-level API details.

## Using Custom Weights

LayerD supports loading custom-trained BiRefNet weights from local paths or remote URLs:

```python
from layerd import LayerD

# Local weights
layerd = LayerD(matting_weight_path="./my_birefnet.pth")

# Remote weights (requires appropriate fsspec backend, e.g., gcsfs for gs://)
layerd = LayerD(matting_weight_path="gs://my-bucket/models/birefnet.pth")
```

## Documentation

- **[Pipeline Guide](docs/pipeline.md)** - High-level API usage and configuration
- **[Export Guide](docs/export.md)** - SVG and PSD export documentation
- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Inference Guide](docs/inference.md)** - Low-level API (CLI and Python)
- **[Training Guide](docs/training.md)** - Training and fine-tuning models
- **[Evaluation Guide](docs/evaluation.md)** - Evaluating layer decomposition quality
- **[Architecture](docs/architecture.md)** - Code architecture and design patterns
- **[Development Guide](docs/development.md)** - Contributing and development workflows
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Contributing](CONTRIBUTING.md)** - How to contribute to LayerD

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

LayerD uses several third-party libraries. See [docs/architecture.md](docs/architecture.md#bundled-dependencies) for details on bundled dependencies.

## Citation

If you find this project useful in your work, please cite our paper.

```bibtex
@inproceedings{suzuki2025layerd,
  title={LayerD: Decomposing Raster Graphic Designs into Layers},
  author={Suzuki, Tomoyuki and Liu, Kang-Jun and Inoue, Naoto and Yamaguchi, Kota},
  booktitle={ICCV},
  year={2025}
}
```
