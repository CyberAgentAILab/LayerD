"""Command-line interface for LayerD layer decomposition."""

import argparse
import logging
import os

from PIL import Image

from layerd.models.layerd import LayerD
from layerd.utils.log import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="LayerD: Decompose raster graphic designs into layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to save layer results",
    )

    # Model configuration
    parser.add_argument(
        "--matting-hf-card",
        type=str,
        default="cyberagent/layerd-birefnet",
        help="HuggingFace model card for matting model (default: cyberagent/layerd-birefnet)",
    )
    parser.add_argument(
        "--matting-process-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Process size for matting model (width height). If not set, uses model's default size.",
    )

    # Inference control
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum iterations for layer decomposition (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the models on (default: cpu)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    return parser.parse_args()


def run_decompose(args: argparse.Namespace) -> None:
    """Run layer decomposition on a single image.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If input file doesn't exist or has invalid format.
        RuntimeError: If device setup fails.
    """
    # Validate input file exists
    if not os.path.isfile(args.input):
        raise ValueError(f"Input file does not exist: {args.input}")

    # Validate file format
    valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
    if not args.input.lower().endswith(valid_extensions):
        raise ValueError(f"Invalid file format. Supported formats: {', '.join(valid_extensions)}")

    # Log configuration
    logger.info("LayerD CLI - Single File Decomposition")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Matting model: {args.matting_hf_card}")
    if args.matting_process_size:
        logger.info(f"Matting process size: {args.matting_process_size[0]}x{args.matting_process_size[1]}")

    # Initialize LayerD model
    logger.info("Initializing LayerD model...")
    layerd = LayerD(
        matting_hf_card=args.matting_hf_card,
        matting_process_size=tuple(args.matting_process_size) if args.matting_process_size else None,
    )

    # Move model to device
    try:
        layerd.to(args.device)
    except Exception as e:
        raise RuntimeError(f"Failed to set device to {args.device}: {e}. Is CUDA available?") from e

    # Load image
    try:
        image = Image.open(args.input)
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}") from e

    # Run decomposition
    logger.info("Processing...")
    layers = layerd.decompose(image, max_iterations=args.max_iterations)

    # Save layers
    filename = os.path.basename(args.input).rsplit(".", 1)[0]
    save_dir = os.path.join(args.output_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    for i, layer in enumerate(layers):
        layer_path = os.path.join(save_dir, f"{i:04d}.png")
        layer.save(layer_path)

    # Log completion
    logger.info(f"Successfully decomposed into {len(layers)} layers")
    logger.info(f"Saved to: {save_dir}/")


def main() -> None:
    """Main entry point for the layerd CLI."""
    args = parse_args()
    setup_logging(level=args.log_level)

    try:
        run_decompose(args)
    except (ValueError, RuntimeError) as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Unexpected error during decomposition: {e}")
        raise


if __name__ == "__main__":
    main()
