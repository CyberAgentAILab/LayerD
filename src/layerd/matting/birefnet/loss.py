from math import exp
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class IoULoss(torch.nn.Module):
    """
    Reference: Mattyus et al., DeepRoadMapper: Extracting Road Topology From Aerial Images, ICCV 2017.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = pred.shape[0]
        IoU = torch.tensor(0.0, device=pred.device)
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)
        # return IoU/b
        return IoU


class PixLoss(nn.Module):
    """
    Pixel loss for each refined map output.
    """

    def __init__(self, lambdas_pix_last: dict[str, float]) -> None:
        super().__init__()
        self.lambdas_pix_last = lambdas_pix_last

        self.criterions_last: dict[str, nn.Module] = {}
        if "bce" in self.lambdas_pix_last and self.lambdas_pix_last["bce"]:
            self.criterions_last["bce"] = nn.BCELoss()
        if "iou" in self.lambdas_pix_last and self.lambdas_pix_last["iou"]:
            self.criterions_last["iou"] = IoULoss()
        if "ssim" in self.lambdas_pix_last and self.lambdas_pix_last["ssim"]:
            self.criterions_last["ssim"] = SSIMLoss()
        if "mae" in self.lambdas_pix_last and self.lambdas_pix_last["mae"]:
            self.criterions_last["mae"] = nn.L1Loss()
        if "mse" in self.lambdas_pix_last and self.lambdas_pix_last["mse"]:
            self.criterions_last["mse"] = nn.MSELoss()

    def forward(
        self, scaled_preds: list[torch.Tensor], gt: torch.Tensor, reduce: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # loss = 0.0
        loss_dict = {}
        for i, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != gt.shape:
                pred_lvl = nn.functional.interpolate(pred_lvl, size=gt.shape[2:], mode="bilinear", align_corners=True)
            for criterion_name, criterion in self.criterions_last.items():
                _loss = criterion(pred_lvl.sigmoid(), gt) * self.lambdas_pix_last[criterion_name]
                # loss += _loss
                loss_dict[f"{criterion_name}_{i}"] = _loss
        return loss_dict if not reduce else sum(loss_dict.values())


class SSIMLoss(torch.nn.Module):
    """
    Reference: Qin et al., BASNet: Boundary-Aware Salient Object Detection, CVPR 2019.
    """

    def __init__(self, window_size: int = 11, size_average: bool = True) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - (1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)) / 2


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def saliency_structure_consistency(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ssim = torch.mean(SSIM(x, y))
    return ssim


LOSS_REGISTRY = {
    "IoULoss": IoULoss,
    "PixLoss": PixLoss,
    "SSIMLoss": SSIMLoss,
}


def build_loss(name: str, **params: Any) -> nn.Module:
    if name in LOSS_REGISTRY:
        return LOSS_REGISTRY[name](**params)
    elif hasattr(torch.nn, name):
        return getattr(torch.nn, name)(**params)
    else:
        raise ValueError(f"{name} is not defined as a loss function.")


def build_loss_cfg(cfg: Any) -> nn.Module:
    return build_loss(cfg.name, **cfg.params)
