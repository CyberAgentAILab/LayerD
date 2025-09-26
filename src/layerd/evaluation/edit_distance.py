from functools import reduce
from typing import Callable

import numpy as np
from PIL import Image

from layerd.data.crello import group_top_layers_indices
from layerd.evaluation import DynamicTimeWarping, build_edit_op, RGBL1, AlphaIoU, RGBL1_AIoU

DEFAULT_SCORE_FNS: dict[str, Callable[[Image.Image, Image.Image], float]] = {"rgb_l1": RGBL1(), "alpha_iou": AlphaIoU()}


class LayersEditDist:
    """Evaluate layer decomposition using edit distance and dynamic time warping."""

    def __init__(
        self,
        cost_fn: Callable[[Image.Image, Image.Image], float] = RGBL1_AIoU(),
        score_fns: dict[str, Callable[[Image.Image, Image.Image], float]] | None = DEFAULT_SCORE_FNS,
        max_edits: int = 5,
        gt_editable: bool = True,
    ) -> None:
        self.max_edits = max_edits
        self.edit_ops = [
            build_edit_op("Merge", layer2_editable=gt_editable, measure_fn=cost_fn, measure_is_similarity=False)
        ]
        self.evaluator = DynamicTimeWarping(cost_fn=cost_fn, score_fns=score_fns)

    def _evaluate_edits(
        self, layers1: list[Image.Image], layers2: list[Image.Image], matched_pairs: list[tuple[int, int]] | None = None
    ) -> tuple[list[list[Image.Image]], list[list[Image.Image]], list[tuple[tuple[int, int] | None, float]], list[str]]:
        tmp_l1s, tmp_l2s, ind_and_scores, editops = [], [], [], []
        # Iterate through available edit operations
        for edit_op in self.edit_ops:
            # Evaluate the edit operation
            tmp_l1, tmp_l2, ind_and_score = edit_op(layers1, layers2, matched_pairs)
            tmp_l1s.append(tmp_l1)
            tmp_l2s.append(tmp_l2)
            ind_and_scores.append(ind_and_score)
            editops.append(edit_op.__class__.__name__)
        return tmp_l1s, tmp_l2s, ind_and_scores, editops

    def _take_edits(
        self,
        tmp_l1s: list[list[Image.Image]],
        tmp_l2s: list[list[Image.Image]],
        ind_and_scores: list[tuple[tuple[int, int] | None, float]],
        editops: list[str],
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        # Find the best edit operation
        best_edit_op = sorted(
            [
                (l1, l2, ind_and_score, op)
                for l1, l2, ind_and_score, op in zip(tmp_l1s, tmp_l2s, ind_and_scores, editops)
            ],
            key=lambda x: x[2][1],  # Sort by score
            reverse=True,
        )[0]  # [0] select top1
        layers1, layers2, _, _ = best_edit_op
        return layers1, layers2

    def _apply_edits(self, layers1: list[Image.Image], layers2: list[Image.Image]) -> list[dict]:
        """Apply edit operations to improve layer matching between two decompositions."""
        results = []

        # Add initial result without edits
        matched_pairs, scores = self.evaluator(layers1, layers2)
        results.append({"edits": 0, **scores})

        # Edit loop
        n_edits = 0
        while n_edits < self.max_edits:
            tmp_l1s, tmp_l2s, ind_and_scores, editops = self._evaluate_edits(layers1, layers2, matched_pairs)

            # Check if any valid edit exists
            if all(score[1] == float("-inf") for score in ind_and_scores):
                break

            layers1, layers2 = self._take_edits(tmp_l1s, tmp_l2s, ind_and_scores, editops)
            n_edits += 1

            # Add result after this edit
            matched_pairs, scores = self.evaluator(layers1, layers2)
            results.append({"edits": n_edits, **scores})

        return results

    def __call__(self, pred_layers: list[Image.Image], gt_layers: list[Image.Image]) -> list[dict]:
        """Evaluate similarity or distance between two layer decompositions with edit operations.

        Args:
            pred_layers: List of predicted PIL images with alpha channel
            gt_layers: List of ground truth PIL images with alpha channel

        Returns:
            list[dict]: List of score dicts for each maximum edit counts (from 0 to max_edits)
        """
        # Grouping using simplified function
        pred_group_indice = group_top_layers_indices(np.array([np.array(l.getchannel("A")) for l in pred_layers]))
        gt_group_indice = group_top_layers_indices(np.array([np.array(l.getchannel("A")) for l in gt_layers]))
        pred_groups = [reduce(Image.alpha_composite, [pred_layers[i] for i in group]) for group in pred_group_indice]
        gt_groups = [reduce(Image.alpha_composite, [gt_layers[i] for i in group]) for group in gt_group_indice]

        # Compute edit distance (returns list of results)
        result = self._apply_edits(pred_groups, gt_groups)

        return result

    @staticmethod
    def aggregate(results: list[list[dict]]) -> list[dict]:
        """Aggregate evaluation results from multiple samples.

        Args:
            results: List of evaluation results, where each result is a list of
                    score dicts for each edit level

        Returns:
            list[dict]: List where each element represents an edit level with:
                - max_edits: The edit level (0 to max_edits)
                - avg_edits: Average of the maximum edits each sample has (capped at this level)
                - avg_score: Average score across all samples
                - count: Number of samples that have this edit level
                - Additional averaged metrics (e.g., avg_alpha_iou, avg_rgb_l1)
        """
        if not results:
            return []

        aggregated = []
        # Find max number of edit levels across all results
        max_edit_levels = max(len(r) for r in results)

        for i in range(max_edit_levels):
            # Get samples that have this edit level
            samples_with_level = [r for r in results if i < len(r)]
            samples_without_level = [r for r in results if i >= len(r)]

            if not samples_with_level:
                continue

            # Calculate average of maximum edits each sample has (capped at current level i)
            max_edits_per_sample = [min(len(r) - 1, i) for r in results]
            edit_stats = {"max_edits": i, "avg_edits": np.mean(max_edits_per_sample), "count": len(samples_with_level)}

            sample_metrics = samples_with_level[0][i].keys()
            for metric in sample_metrics:
                # Collect values from samples that have this level
                values = [r[i][metric] for r in samples_with_level]
                # Add best scores from samples that don't have this level
                values.extend([r[-1][metric] for r in samples_without_level])

                edit_stats[f"avg_{metric}"] = np.mean(values)

            aggregated.append(edit_stats)

        return aggregated
