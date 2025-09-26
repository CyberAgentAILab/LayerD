from typing import Any, Callable

from PIL import Image


class EditOp:
    def __init__(
        self,
        measure_fn: Callable[[Image.Image, Image.Image], float],
        layer2_editable: bool = False,
        min_layers: int = 2,
        measure_is_similarity: bool = True,
    ) -> None:
        self.min_layers = min_layers
        self.layer2_editable = layer2_editable
        self.measure_is_similarity = measure_is_similarity
        if not measure_is_similarity:
            self.sim_fn = lambda x, y: 1 - measure_fn(x, y)
        else:
            self.sim_fn = measure_fn

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class Merge(EditOp):
    def __call__(
        self,
        layers1: list[Image.Image],
        layers2: list[Image.Image],
        matched_pairs: list[tuple[int, int]] | None = None,
    ) -> tuple[list[Image.Image], list[Image.Image], tuple[tuple[int, int] | None, float]]:
        """Find the best merge operation to improve layer matching.

        Args:
            layers1: List of predicted layer images to potentially merge
            layers2: List of ground truth layer images
            matched_pairs: Optional list of (layer1_idx, layer2_idx) tuples indicating
                          which layers are matched. If None, assumes 1-to-1 matching.

        Returns:
            Tuple of (modified_layers1, modified_layers2, best_match) where:
            - modified_layers1: layers1 with best merge applied (or original if no good merge)
            - modified_layers2: layers2 with best merge applied if layer2_editable=True
            - best_match: Tuple of (merge_indices, score_gain) where:
                * merge_indices: Tuple of (i, i+1) indices that were merged, or None
                * score_gain: The improvement in matching score from merging (higher is better)
        """
        l1_merged_idx, l1_merged_score_diff = self._compute_merged_scores(layers1, layers2, matched_pairs)
        best_l1_matched = self._find_best_match(l1_merged_idx, l1_merged_score_diff)
        best_l2_matched: tuple[tuple[int, int] | None, float] = (None, float("-inf"))
        if self.layer2_editable and matched_pairs is not None:
            l2_merged_idx, l2_merged_score_diff = self._compute_merged_scores(
                layers2, layers1, [(j, i) for i, j in matched_pairs]
            )
            best_l2_matched = self._find_best_match(l2_merged_idx, l2_merged_score_diff)
        return self._apply_best_merge(layers1, layers2, best_l1_matched, best_l2_matched)

    def _compute_merged_scores(
        self, ls: list[Image.Image], gts: list[Image.Image], matched_pairs: list[tuple[int, int]] | None = None
    ) -> tuple[list[tuple[int, int]], list[float]]:
        merged_ids: list[tuple[int, int]] = []
        gains: list[float] = []
        if len(ls) <= self.min_layers:
            return merged_ids, gains

        for i in range(len(ls) - 1):
            # Step 1: Compute merged candidate for layers i and i+1
            merged_candidate = self._merge(ls[i], ls[i + 1])
            # Candidate list: first candidate is the merged layer, and if available second candidate is ls[i+2]
            subls = [merged_candidate]
            if i + 2 < len(ls):
                subls.append(ls[i + 2])

            # Step 2: Gather corresponding ground truth layers for layers i and i+1
            if matched_pairs is not None:
                subgts = [
                    [gts[p[1]] for p in matched_pairs if p[0] == i],
                    [gts[p[1]] for p in matched_pairs if p[0] == i + 1],
                ]
            else:
                subgts = [[gts[i]], [gts[i + 1]]]

            # Step 3: Compute current distance sum for layers i and i+1
            curD = sum(self.sim_fn(ls[i], gt) for gt in subgts[0]) + sum(self.sim_fn(ls[i + 1], gt) for gt in subgts[1])

            # Step 4: For each candidate in subls, compute candidate distance sum over both groups.
            candidateDs = []
            for candidate in subls:
                for group in subgts:
                    candidateDs.append(sum(self.sim_fn(candidate, gt) for gt in group))

            if len(candidateDs) > 1:
                adjustedDs = [d + candidateDs[0] for d in candidateDs[1:]]
                maxD = max(adjustedDs)
            else:
                maxD = candidateDs[0]

            # Step 5: If merging reduces the current distance, record the index and gain.
            if maxD > curD:
                merged_ids.append((i, i + 1))
                gains.append(maxD - curD)
        return merged_ids, gains

    def _find_best_match(
        self, merged_idx: list[tuple[int, int]], merged_score_diff: list[float]
    ) -> tuple[tuple[int, int] | None, float]:
        if not merged_score_diff:
            return (None, float("-inf"))
        return sorted(zip(merged_idx, merged_score_diff), key=lambda x: x[1], reverse=True)[0]

    def _apply_best_merge(
        self,
        layers1: list[Image.Image],
        layers2: list[Image.Image],
        best_l1: tuple[tuple[int, int] | None, float],
        best_l2: tuple[tuple[int, int] | None, float],
    ) -> tuple[list[Image.Image], list[Image.Image], tuple[tuple[int, int] | None, float]]:
        best_match: tuple[tuple[int, int] | None, float] = (None, float("-inf"))
        if best_l1 == (None, float("-inf")) and self.layer2_editable and best_l2 == (None, float("-inf")):
            return layers1, layers2, best_match
        if best_l1[1] > best_l2[1]:
            best_match = best_l1
            if best_match[0] is not None:
                merged = self._merge(layers1[best_match[0][0]], layers1[best_match[0][1]])
                layers1.pop(best_match[0][1])
                layers1.pop(best_match[0][0])
                layers1.insert(best_match[0][0], merged)
        else:
            if best_l2 != (None, float("-inf")):
                best_match = best_l2
                if best_match[0] is not None:
                    merged = self._merge(layers2[best_match[0][0]], layers2[best_match[0][1]])
                    layers2.pop(best_match[0][1])
                    layers2.pop(best_match[0][0])
                    layers2.insert(best_match[0][0], merged)
        return layers1, layers2, best_match

    def _merge(self, l1: Image.Image, l2: Image.Image) -> Image.Image:
        return Image.alpha_composite(l1, l2)  # l2 over l1


# Registry for edit operations
EDIT_OPS = {
    "Merge": Merge,
}


def build_edit_op(name: str, **kwargs: Any) -> EditOp:
    """Create an edit operation by name.

    Args:
        name: Name of the edit operation (e.g., "Merge")
        **kwargs: Additional arguments for operation initialization

    Returns:
        Initialized edit operation instance

    Raises:
        ValueError: If the operation name is not found in registry
    """
    if name not in EDIT_OPS:
        available = list(EDIT_OPS.keys())
        raise ValueError(f"Unknown edit operation: {name}. Available: {available}")
    return EDIT_OPS[name](**kwargs)
