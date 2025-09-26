from typing import Callable

import numpy as np
from PIL import Image


def dynamic_time_warping(
    pred_images: list[Image.Image],
    gt_images: list[Image.Image],
    cost_fn: Callable[[Image.Image, Image.Image], float],
    step: int = 1,
    const: float = 0,
    cost_range: tuple[float, float] | None = (0, 1),
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Compute Dynamic Time Warping alignment between predicted and ground truth images.

    Args:
        pred_images: List of predicted PIL images
        gt_images: List of ground truth PIL images
        cost_fn: Cost function for comparing images
        step: Step size for DTW path (default=1)
        const: Constant value to add to the cost (default=0)
        cost_range: Optional tuple specifying valid range for cost values (min, max)

    Returns:
        Tuple containing:
            - path: DTW alignment path as list of (pred_idx, gt_idx) tuples
            - C: Cost matrix
            - D: DTW distance matrix
    """
    n, m = len(pred_images), len(gt_images)

    # Compute cost matrix
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            _s = cost_fn(pred_images[i], gt_images[j])
            if cost_range is not None:
                assert cost_range[0] <= _s <= cost_range[1], f"Cost {_s} out of range {cost_range}"
            C[i, j] = _s + const

    def _cum(C: np.ndarray) -> np.ndarray:
        """Compute accumulated cost matrix D from cost matrix C."""
        n, m = C.shape
        D = np.zeros((n, m))
        D[0, 0] = C[0, 0]
        for i in range(1, n):
            D[i, 0] = D[i - 1, 0] + C[i, 0]
        for j in range(1, m):
            D[0, j] = D[0, j - 1] + C[0, j]
        for i in range(1, n):
            for j in range(1, m):
                D[i, j] = C[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return D

    # Compute accumulated cost matrix
    D = _cum(C)
    i, j = n - 1, m - 1
    path = [(i, j)]
    while True:
        if i <= 0 and j <= 0:
            break
        # Boundary conditions
        elif i == 0:
            path.append((i, j - step))
            j -= step
        elif j == 0:
            path.append((i - step, j))
            i -= step
        elif D[i - step, j - step] <= D[i - step, j] and D[i - step, j - step] <= D[i, j - step]:
            path.append((i - step, j - step))
            i -= step
            j -= step
        elif D[i - step, j] <= D[i - step, j - step] and D[i - step, j] <= D[i, j - step]:
            path.append((i - step, j))
            i -= step
        else:
            path.append((i, j - step))
            j -= step
    path.reverse()
    return path, C, D


class DynamicTimeWarping:
    def __init__(
        self,
        cost_fn: Callable[[Image.Image, Image.Image], float],
        score_fns: dict[str, Callable[[Image.Image, Image.Image], float]] | None = None,
        vis: bool = False,
    ) -> None:
        """
        Args:
            cost_fn: Cost function for comparing images
            score_fns: Optional dict of additional scoring functions
            vis: Whether to generate visualization of DTW alignment
        """
        self.cost_fn = cost_fn
        self.score_fns = score_fns if score_fns is not None else {}
        self.vis = vis

    @staticmethod
    def calc_score(
        func: Callable[[Image.Image, Image.Image], float],
        path: list[tuple[int, int]],
        pred_images: list[Image.Image],
        gt_images: list[Image.Image],
    ) -> float:
        """Calculate scores for DTW-aligned image pairs."""
        results = []
        for i, j in path:
            result = func(pred_images[i], gt_images[j])
            results.append(result)

        return float(np.mean(results))

    def __call__(
        self, pred_images: list[Image.Image], gt_images: list[Image.Image]
    ) -> tuple[list[tuple[int, int]], dict[str, float]]:
        """
        Args:
            pred_images: List of predicted PIL images
            gt_images: List of ground truth PIL images
        Returns:
            Tuple containing:
            - path: DTW alignment path as list of (pred_idx, gt_idx) tuples
            - scores: Dictionary of computed scores for the aligned image pairs
        """
        # Compute DTW alignment
        path, C, D = dynamic_time_warping(pred_images, gt_images, cost_fn=self.cost_fn)

        # Compute scores on aligned pairs
        scores = {"cost": self.calc_score(self.cost_fn, path, pred_images, gt_images)}
        scores.update({name: self.calc_score(fn, path, pred_images, gt_images) for name, fn in self.score_fns.items()})

        return path, scores
