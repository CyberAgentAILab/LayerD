import logging
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForImageSegmentation

from .dataset import MattingDataset

# Suppress timm warnings from BiRefNet's HuggingFace model code
# The remote model uses deprecated timm imports (timm.models.layers -> timm.layers)
# This is a temporary fix until the upstream model is updated
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.registry")


logger = logging.getLogger(__name__)


def build_birefnet(card: str = "cyberagent/layerd-birefnet", **params: Any) -> nn.Module:
    return AutoModelForImageSegmentation.from_pretrained(card, trust_remote_code=True, **params)


def worker_init_fn(worker_id: int) -> None:
    state = np.random.get_state()
    seed = int(state[1][0]) + worker_id  # type: ignore[index]
    np.random.seed(seed)


def build_dataloader_cfg(cfg: Any, distributed: bool = False) -> torch.utils.data.DataLoader:
    dataset = MattingDataset(**cfg.params)
    sampler: DistributedSampler | None = None if not distributed else DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
    )
    return dataloader
