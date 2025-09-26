import argparse
import json
import logging
import os
import os.path as osp
from glob import glob

from PIL import Image
from tqdm import tqdm

from layerd.evaluation import LayersEditDist
from layerd.utils.log import setup_logging

logger = logging.getLogger(__name__)


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate layer decomposition results using LayersEditDist."""

    os.makedirs(args.output_dir, exist_ok=True)
    evaluator = LayersEditDist(max_edits=args.max_edits)

    output_paths = sorted([p for p in glob(osp.join(args.pred_dir, "*")) if osp.isdir(p)])
    results = []

    for path in tqdm(output_paths, desc="Evaluating results", ncols=0):
        fn = osp.basename(path)
        logger.info(f"Processing {fn}")
        if not osp.exists(osp.join(args.gt_dir, fn)):
            logger.warning(f"Skipping {fn} as no ground truth directory found")
            continue
        layers = [Image.open(p).convert("RGBA") for p in sorted(glob(osp.join(path, "*.png")))]
        layers_gt = [Image.open(p).convert("RGBA") for p in sorted(glob(osp.join(args.gt_dir, fn, "*.png")))]
        if len(layers_gt) < 2 or len(layers) < 2:
            logger.warning(f"Skipping {fn} due to less than 2 layers in {'pred' if len(layers) < 2 else 'gt'}")
            continue

        result = evaluator(layers, layers_gt)
        with open(osp.join(args.output_dir, f"{fn}.json"), "w") as f:
            json.dump(result, f, indent=2)
        results.append(result)

    aggregated_scores = LayersEditDist.aggregate(results)
    with open(osp.join(args.output_dir, "aggregated_scores.json"), "w") as f:
        json.dump(aggregated_scores, f, indent=2)
    logger.info(f"Aggregated scores:\n{json.dumps(aggregated_scores, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate layer decomposition results")
    parser.add_argument("--pred-dir", type=str, help="Directory with predicted layer decompositions")
    parser.add_argument("--gt-dir", type=str, help="Directory with ground truth layer decompositions")
    parser.add_argument("--output-dir", type=str, help="Directory to save evaluation results")
    parser.add_argument("--max-edits", type=int, default=5, help="Maximum number of edit operations to apply")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level, use_tqdm_handler=True)
    evaluate(args)
