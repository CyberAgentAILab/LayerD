import logging
import os
import os.path as osp
import sys

from hydra import compose, initialize_config_dir

from layerd.matting.birefnet.train import train as train_birefnet
from layerd.utils.log import setup_logging

logger = logging.getLogger(__name__)


def parse_config_path(args: list[str]) -> tuple[str, list[str]]:
    config_paths = [a for a in args if a.startswith("config_path=")]
    assert len(config_paths) == 1, "'config_path' should be specified, like 'config_path=...'"
    args.pop(args.index(config_paths[0]))
    config_path = config_paths[0].split("=")[-1]
    config_path = osp.abspath(config_path)
    return config_path, args


def train() -> None:
    args = sys.argv

    # Extract log level if provided
    log_level = "INFO"
    log_args = [a for a in args if a.startswith("log_level=")]
    if log_args:
        log_level = log_args[0].split("=")[-1]
        args.pop(args.index(log_args[0]))

    setup_logging(level=log_level, use_tqdm_handler=True)

    config_path, args = parse_config_path(args)
    config_dir, config_name = osp.split(config_path)
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=args[1:])

    if cfg.model.name == "birefnet":
        train_birefnet(cfg)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


if __name__ == "__main__":
    train()
