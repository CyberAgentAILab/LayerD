import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


class MAE:
    """Mean Absolute Error metric."""

    def __init__(self, normalize_value: float = 1.0) -> None:
        self.normalize_value = normalize_value

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x / self.normalize_value
        y = y / self.normalize_value
        return np.mean(np.abs(x - y))


class SSIM:
    """Structural Similarity Index metric."""

    def __init__(self, data_range: float = 1, normalize_value: float = 1.0) -> None:
        self.data_range = data_range
        self.normalize_value = normalize_value

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x / self.normalize_value
        y = y / self.normalize_value
        values = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                values.append(ssim(x[i, j], y[i, j], data_range=self.data_range))
        return np.mean(values)


class RGBL1:
    """RGB L1 distance metric with optional alpha weighting."""

    def __init__(self, normalize_value: float = 255, weight_by_alpha: bool = True, eps: float = 1e-8) -> None:
        self.normalize_value = normalize_value
        self.weight_by_alpha = weight_by_alpha
        self.eps = eps

    def __call__(self, rgba_pr: np.ndarray | Image.Image, rgba_gt: np.ndarray | Image.Image) -> float:
        """Calculate RGB L1 distance between RGBA images.

        Args:
            rgba_pr: Predicted RGBA image (H, W, 4)
            rgba_gt: Ground truth RGBA image (H, W, 4)

        Returns:
            RGB L1 distance value
        """
        assert isinstance(rgba_pr, Image.Image) or (isinstance(rgba_pr, np.ndarray) and rgba_pr.dtype == np.uint8)
        assert isinstance(rgba_gt, Image.Image) or (isinstance(rgba_gt, np.ndarray) and rgba_gt.dtype == np.uint8)

        rgba_pr = np.array(rgba_pr).astype(float) / self.normalize_value
        rgba_gt = np.array(rgba_gt).astype(float) / self.normalize_value

        if self.weight_by_alpha:
            alpha_mask = rgba_gt[..., 3] > 0
            return (np.abs(rgba_pr[..., :3] - rgba_gt[..., :3]).mean(-1) * alpha_mask).sum() / (
                alpha_mask.sum() + self.eps
            )
        else:
            return np.abs(rgba_pr[..., :3] - rgba_gt[..., :3]).mean()


def calc_soft_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate soft IoU between two masks."""
    intersection = np.minimum(mask1, mask2).sum()
    union = np.maximum(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


class AlphaIoU:
    """Alpha channel Intersection over Union metric."""

    def __init__(self, normalize_value: float = 255) -> None:
        self.normalize_value = normalize_value

    def __call__(self, rgba_pr: np.ndarray | Image.Image, rgba_gt: np.ndarray | Image.Image) -> float:
        """Calculate IoU between alpha channels.

        Args:
            rgba_pr: Predicted RGBA image (H, W, 4)
            rgba_gt: Ground truth RGBA image (H, W, 4)

        Returns:
            Alpha channel IoU value between 0 and 1
        """
        assert isinstance(rgba_pr, Image.Image) or (isinstance(rgba_pr, np.ndarray) and rgba_pr.dtype == np.uint8)
        assert isinstance(rgba_gt, Image.Image) or (isinstance(rgba_gt, np.ndarray) and rgba_gt.dtype == np.uint8)

        rgba_pr = np.array(rgba_pr).astype(float) / self.normalize_value
        rgba_gt = np.array(rgba_gt).astype(float) / self.normalize_value

        soft_iou = calc_soft_iou(rgba_pr[..., 3], rgba_gt[..., 3])

        return soft_iou


class RGBL1_AIoU:
    """Combined RGB L1 and Alpha IoU metric."""

    def __init__(self, alpha_weight: float = 1, normalize_value: float = 255) -> None:
        self._rgb_l1 = RGBL1(normalize_value=normalize_value, weight_by_alpha=True)
        self._alpha_iou = AlphaIoU(normalize_value=normalize_value)
        self._rgb_l1_weight = 1
        self._alpha_iou_weight = alpha_weight

    def __call__(self, rgba_pr: np.ndarray | Image.Image, rgba_gt: np.ndarray | Image.Image) -> float:
        """Calculate combined RGB L1 and Alpha IoU metric.

        Args:
            rgba_pr: Predicted RGBA image (H, W, 4)
            rgba_gt: Ground truth RGBA image (H, W, 4)

        Returns:
            Combined metric score (lower is better)
        """
        rgb_l1 = self._rgb_l1(rgba_pr, rgba_gt)
        alpha_iou = self._alpha_iou(rgba_pr, rgba_gt)
        score = (self._rgb_l1_weight * rgb_l1 + self._alpha_iou_weight * (1 - alpha_iou)) / (
            self._rgb_l1_weight + self._alpha_iou_weight
        )
        return score


METRICS = {
    "MAE": MAE,
    "SSIM": SSIM,
    "RGBL1": RGBL1,
    "AlphaIoU": AlphaIoU,
    "RGBL1_AIoU": RGBL1_AIoU,
}


def build_metric_cfg(name: str, params: dict) -> object:
    """Build a single metric instance from name and parameters."""
    if name in METRICS:
        return METRICS[name](**params)
    else:
        raise ValueError(f"Unknown metric: {name}")


def build_metrics_cfg(cfg: dict) -> dict:
    """Build metrics configuration from dictionary."""
    metrics = {}
    for name, params in cfg.items():
        metrics[name] = build_metric_cfg(name, params)
    return metrics
