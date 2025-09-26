import argparse
import logging
import os
import os.path as osp

import datasets
import huggingface_hub
from PIL import Image
from tqdm import tqdm

from layerd.data.crello import make_top_layers_alpha_and_image_pair
from layerd.data.renderer import CrelloV5RendererLayers
from layerd.models.inpaint import LamaInpaint
from layerd.utils.log import setup_logging

logger = logging.getLogger(__name__)


def generate_crello_matting(args: argparse.Namespace) -> None:
    dataset_name = "cyberagent/crello"
    dataset = datasets.load_dataset(dataset_name, revision=args.revision, cache_dir=args.hf_cache_dir)
    fonts_path = huggingface_hub.hf_hub_download(
        repo_id=dataset_name, revision=args.revision, filename="resources/fonts.pickle", repo_type="dataset"
    )
    renderer = CrelloV5RendererLayers(dataset["train"].features, fonts_path)
    inpaint_model = LamaInpaint(device=args.device) if args.inpainting else None

    for split in args.splits:
        os.makedirs(osp.join(args.output_dir, split, "im"), exist_ok=True)
        os.makedirs(osp.join(args.output_dir, split, "gt"), exist_ok=True)

        start_i = 0 if args.start_index is None else max(args.start_index, 0)
        end_i = len(dataset[split]) if args.end_index is None else min(args.end_index, len(dataset[split]))
        ds = dataset[split].select(range(start_i, end_i))
        n_ds = len(ds) if args.num_samples < 0 else min(len(ds), args.num_samples)

        for idx, example in enumerate(tqdm(ds, ncols=0, desc=f"Generating dataset ({split})", total=n_ds)):
            sample_i = idx + start_i
            logger.info(f"Processing {sample_i}/{n_ds}: {example['id']}")
            pairs, inpainted_pairs, layers_pil = make_top_layers_alpha_and_image_pair(
                example,
                renderer,
                short_side_size=args.short_side_size,
                exclude_text=args.exclude_text,
                exclude_transparent=True,  # LayerD supports only opaque layers
                inpaint_model=inpaint_model,
            )
            # Save training pairs
            for j, (fg_mask, image) in enumerate(pairs):
                fg_mask_pil = Image.fromarray(fg_mask)
                fg_mask_pil.save(
                    osp.join(args.output_dir, split, "gt", f"{split}_{sample_i:06d}_{example['id']}_{j:02d}.png")
                )
                image.save(
                    osp.join(args.output_dir, split, "im", f"{split}_{sample_i:06d}_{example['id']}_{j:02d}.png")
                )
            for j, pair in enumerate(inpainted_pairs):
                if pair is None:
                    continue
                fg_mask, image = pair
                fg_mask_pil = Image.fromarray(fg_mask)
                fg_mask_pil.save(
                    osp.join(
                        args.output_dir, split, "gt", f"{split}_{sample_i:06d}_{example['id']}_{j:02d}_inpainted.png"
                    )
                )
                image.save(
                    osp.join(
                        args.output_dir, split, "im", f"{split}_{sample_i:06d}_{example['id']}_{j:02d}_inpainted.png"
                    )
                )
            if args.save_layers and len(layers_pil) > 0:
                os.makedirs(osp.join(args.output_dir, split, "composite"), exist_ok=True)
                composite_image = pairs[0][1]
                composite_image.save(
                    osp.join(args.output_dir, split, "composite", f"{split}_{sample_i:06d}_{example['id']}.png")
                )
                out_dir = osp.join(args.output_dir, split, "layers", f"{split}_{sample_i:06d}_{example['id']}")
                os.makedirs(osp.join(out_dir), exist_ok=True)
                for j, layer_pil in enumerate(layers_pil):
                    layer_pil.save(osp.join(out_dir, f"{split}_{sample_i:06d}_{example['id']}_{j:02d}.png"))

            if (idx + 1) == args.num_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="Huggingface cache dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save results")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"], help="Dataset splits")
    parser.add_argument("--short-side-size", type=int, default=512, help="Short side size for rendering")
    parser.add_argument("--exclude-text", action="store_true", help="Exclude text layers")
    parser.add_argument("--inpainting", action="store_true", help="Use inpainting to fill occluded regions")
    parser.add_argument("--save-layers", action="store_true", help="Save filtered layers and composite images")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the inpainting model on")
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples to process per split")
    parser.add_argument("--start-index", type=int, default=None, help="Start index to process")
    parser.add_argument("--end-index", type=int, default=None, help="End index to process")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument("--revision", type=str, default="5.1.0", help="Crello dataset version")
    args = parser.parse_args()

    setup_logging(level=args.log_level, use_tqdm_handler=True)
    logger.info(f"Arguments: {args}")
    generate_crello_matting(args)
