import datetime
import logging
import os
import os.path as osp
import pprint
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from layerd.evaluation import build_metrics_cfg
from layerd.utils import read_json, save_json, save_yaml, set_seed

from . import build_birefnet, build_dataloader_cfg
from .loss import PixLoss, build_loss_cfg

logger = logging.getLogger(__name__)


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        x /= torch.distributed.get_world_size()
    return x


def _step_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_fns: dict[str, Any],
    optimizer: optim.Optimizer,
    metric_fns: Any | None = None,
    is_train: bool = True,
    out_ref: bool = True,
    device: str | torch.device = "cpu",
    accelerator: Any | None = None,
) -> dict[str, float]:
    if accelerator is not None:
        inputs = batch[0]
        gts = batch[1]
    else:
        inputs = batch[0].to(device)
        gts = batch[1].to(device)
    scaled_preds = model(inputs)[0] if is_train else model(inputs)
    if out_ref and is_train:
        (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
        loss_gdt_dict = {}
        for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
            _gdt_pred = nn.functional.interpolate(
                _gdt_pred,
                size=_gdt_label.shape[2:],
                mode="bilinear",
                align_corners=True,
            ).sigmoid()
            _gdt_label = _gdt_label.sigmoid()
            loss_gdt_dict[f"gdt_{_idx}"] = loss_fns["gdt_loss"](_gdt_pred, _gdt_label)
        loss_gdt = sum(loss_gdt_dict.values())

    loss_dict = loss_fns["pix_loss"](scaled_preds, torch.clamp(gts, 0, 1), reduce=False)
    loss = sum(loss_dict.values())
    if out_ref and is_train:
        loss += loss_gdt * 1.0

    if is_train:
        optimizer.zero_grad()
        if accelerator is not None:
            loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(loss)
        else:
            loss.backward()
        optimizer.step()
    res = {"loss": all_reduce_mean(loss).item()}
    res.update({key: all_reduce_mean(value).item() for key, value in loss_dict.items()})
    if out_ref and is_train:
        res.update({key: value.item() for key, value in loss_gdt_dict.items()})

    if metric_fns is not None:
        metrics_values = {}
        for key, fn in metric_fns.items():
            value = fn(scaled_preds[-1].sigmoid().detach().cpu().numpy(), gts.detach().cpu().numpy())
            metrics_values[key] = all_reduce_mean(torch.tensor(value, device=device)).item()
        res.update(metrics_values)
    return res


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fns: dict[str, Any],
    optimizer: optim.Optimizer,
    metric_fns: Any,
    out_ref: bool = True,
    device: str | torch.device = "cpu",
    accelerator: Any | None = None,
) -> dict[str, float]:
    model.train()
    metrics_list = []
    device = device if isinstance(device, torch.device) else torch.device(device)
    rank = 0 if getattr(device, "index", None) is None else device.index
    for batch in tqdm(data_loader, total=len(data_loader), desc=f"Training ({device})", ncols=0, disable=rank != 0):
        res_batch = _step_batch(
            model,
            batch,
            loss_fns,
            optimizer,
            metric_fns,
            is_train=True,
            out_ref=out_ref,
            device=device,
            accelerator=accelerator,
        )
        metrics_list.append(res_batch)
    metrics_avg = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0].keys()}
    return metrics_avg


def test_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fns: dict[str, Any],
    optimizer: optim.Optimizer,  # Required for API consistency but not used during validation
    metric_fns: Any,
    out_ref: bool = True,
    device: str | torch.device = "cpu",
    accelerator: Any | None = None,
) -> dict[str, float]:
    model.eval()
    metrics_list = []
    device = device if isinstance(device, torch.device) else torch.device(device)
    rank = 0 if getattr(device, "index", None) is None else device.index
    with torch.no_grad():
        for batch in tqdm(
            data_loader, total=len(data_loader), desc=f"Validation ({device})", ncols=0, disable=rank != 0
        ):
            res_batch = _step_batch(
                model=model,
                batch=batch,
                loss_fns=loss_fns,
                optimizer=optimizer,  # Pass optimizer even though not used
                metric_fns=metric_fns,
                is_train=False,
                out_ref=out_ref,
                device=device,
                accelerator=accelerator,
            )
            metrics_list.append(res_batch)
    metrics_avg = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0].keys()}
    return metrics_avg


def save_all_states(
    model: nn.Module | DDP,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    out_dir: str,
    suffix: str,
) -> None:
    weight = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(weight, os.path.join(out_dir, f"model_{suffix}.pth"))
    torch.save(optimizer.state_dict(), os.path.join(out_dir, f"optimizer_{suffix}.pth"))
    torch.save(scheduler.state_dict(), os.path.join(out_dir, f"lr_scheduler_{suffix}.pth"))
    logger.info(f"Saved model, optimizer, and scheduler states to {out_dir} with suffix {suffix}")


def build_optimizer(model: nn.Module, optimizer_cfg: Any) -> optim.Optimizer:
    """Build optimizer from configuration."""
    if hasattr(optim, optimizer_cfg.name):
        return getattr(optim, optimizer_cfg.name)(model.parameters(), **optimizer_cfg.params)
    else:
        raise NotImplementedError(f"{optimizer_cfg.name} is not implemented for optimizer")


def build_scheduler(optimizer: optim.Optimizer, scheduler_cfg: Any) -> lr_scheduler.LRScheduler:
    """Build learning rate scheduler from configuration."""
    if scheduler_cfg.name == "off":
        return lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    elif hasattr(lr_scheduler, scheduler_cfg.name):
        return getattr(lr_scheduler, scheduler_cfg.name)(optimizer, **scheduler_cfg.params)
    else:
        raise NotImplementedError(f"{scheduler_cfg.name} is not implemented for scheduler")


def train(cfg: Any) -> None:
    set_seed(cfg.seed)
    accelerator = None
    distributed = cfg.dist
    if cfg.use_accelerate:
        from accelerate import Accelerator

        accelerator = Accelerator(
            mixed_precision=cfg.mixed_precision,
            gradient_accumulation_steps=1,
        )
        distributed = False

    if distributed:
        init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 10))
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0
    else:
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0 if cfg.use_accelerate else True
    if is_main_process:
        cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        os.makedirs(cfg.out_dir, exist_ok=True)
        save_yaml(cfg_dict, osp.join(cfg.out_dir, "config.yaml"))
        logger.info(f"Start matting model training with config:\n{pprint.pformat(cfg_dict, indent=2)}")

    # Model, Optimizer, Scheduler
    model = build_birefnet(cfg.model.card, **cfg.model.params)
    if cfg.resume:
        model.load_state_dict(
            torch.load(os.path.join(cfg.out_dir, "model_last.pth"), map_location="cpu", weights_only=True)
        )
    if not cfg.use_accelerate:
        if distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        else:
            model = model.to(device)
    if cfg.compile:
        model = cast(nn.Module | DDP, torch.compile(model, mode=cfg.compile))

    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler)

    if cfg.resume:
        optimizer.load_state_dict(
            torch.load(os.path.join(cfg.out_dir, "optimizer_last.pth"), map_location="cpu", weights_only=True)
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(cfg.out_dir, "lr_scheduler_last.pth"), map_location="cpu", weights_only=True)
        )

    # Dataloader
    train_loader = build_dataloader_cfg(cfg.dataset.train, distributed=distributed)
    val_loader = build_dataloader_cfg(cfg.dataset.val, distributed=distributed)

    # Prepare with accelerator if using it
    if accelerator is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
        device = accelerator.device

    # Loss
    pix_loss = cast(PixLoss, build_loss_cfg(cfg.loss.pix_loss))
    gdt_loss = build_loss_cfg(cfg.loss.gdt_loss)
    loss_fns = {"pix_loss": pix_loss, "gdt_loss": gdt_loss}

    # Metrics
    metric_fns = build_metrics_cfg(cfg.metrics)

    all_metrics: list[dict[str, Any]] = []
    epoch_start = 1
    if cfg.resume:
        all_metrics = read_json(os.path.join(cfg.out_dir, "metrics.json"))  # type: ignore[assignment]
        epoch_start = max([m["epoch"] for m in all_metrics]) + 1

    if is_main_process:
        logger.info(f"Starting training from epoch {epoch_start}")

    # Training loop
    for epoch in range(epoch_start, cfg.epochs + 1):
        if epoch > cfg.epochs + cfg.finetune_last_epochs:
            if cfg.is_matting:
                pix_loss.lambdas_pix_last["mae"] *= 1
                pix_loss.lambdas_pix_last["mse"] *= 0.9
                pix_loss.lambdas_pix_last["ssim"] *= 0.9
            else:
                pix_loss.lambdas_pix_last["bce"] *= 0
                pix_loss.lambdas_pix_last["ssim"] *= 1
                pix_loss.lambdas_pix_last["iou"] *= 0.5
                pix_loss.lambdas_pix_last["mae"] *= 0.9
        loss_fns = {"pix_loss": pix_loss, "gdt_loss": gdt_loss}
        res_train = train_epoch(
            model=model,
            data_loader=train_loader,
            loss_fns=loss_fns,
            optimizer=optimizer,
            metric_fns=metric_fns,
            out_ref=True,
            device=device,
            accelerator=accelerator,
        )
        res_val = {}
        if epoch % cfg.valid_step == 0:
            res_val = test_epoch(
                model=model,
                data_loader=val_loader,
                loss_fns=loss_fns,
                optimizer=optimizer,  # Pass optimizer to test_epoch
                metric_fns=metric_fns,
                out_ref=True,
                device=device,
                accelerator=accelerator,
            )
        if is_main_process:
            all_metrics.append({"epoch": epoch, **res_train, **res_val})
            save_json(all_metrics, os.path.join(cfg.out_dir, "metrics.json"))
            if epoch % cfg.save_step == 0:
                save_all_states(model, optimizer, scheduler, cfg.out_dir, f"epoch_{epoch:03}")
            save_all_states(model, optimizer, scheduler, cfg.out_dir, "last")

        if cfg.use_accelerate or distributed:
            torch.distributed.barrier()

    if distributed:
        destroy_process_group()
