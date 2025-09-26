import io
import logging

import numpy as np
from PIL import Image

from layerd.data.renderer import CrelloV5RendererLayers
from layerd.models.helpers import expand_mask, shrink_mask
from layerd.models.inpaint import BaseInpaint

logger = logging.getLogger(__name__)


def group_top_layers_indices(alpha_layers: np.ndarray) -> list[list[int]]:
    """Group layers into non-overlapping sets based on occlusion.

    Args:
        alpha_layers: Array of shape (n_layers, H, W) with alpha channels.

    Returns:
        List of layer index groups ordered back-to-front.
    """
    top_layers_indices_groups = []
    remaining_indices = list(range(len(alpha_layers)))
    hard_mask = alpha_layers > 0
    while remaining_indices:
        current_group = []
        for i, target_index in enumerate(remaining_indices):  # Back-to-front
            is_top_layer = True
            for _, compare_index in enumerate(remaining_indices[i + 1 :]):
                overlap = (hard_mask[target_index] & hard_mask[compare_index]).sum()
                if overlap > 0:
                    is_top_layer = False
                    break
            if is_top_layer:
                current_group.append(target_index)
        top_layers_indices_groups.append(current_group)
        remaining_indices = [i for i in remaining_indices if i not in current_group]  # Pop current group
    return top_layers_indices_groups[::-1]  # Reverse to have back-to-front


def make_top_layers_alpha_and_image_pair(
    example: dict,
    renderer: CrelloV5RendererLayers,
    short_side_size: int = 360,
    exclude_text: bool = False,
    exclude_transparent: bool = False,
    inpaint_model: BaseInpaint | None = None,
) -> tuple[list[tuple[np.ndarray, Image.Image]], list[tuple[np.ndarray, Image.Image]], list[Image.Image]]:
    """Create alpha mask and image pairs for layer decomposition training.

    Args:
        example: Crello dataset example containing design data.
        renderer: Renderer for generating layer images.
        short_side_size: Target size for the shorter image dimension.
        exclude_text: If True, exclude text layers.
        exclude_transparent: If True, exclude transparent layers.
        inpaint_model: Optional model for inpainting occluded regions.

    Returns:
        - pairs: List of (alpha_mask, image) tuples ordered front-to-back
        - inpainted_pairs: List of (alpha_mask, image) tuples with inpainted regions
        - layers_pil: List of filtered PIL Image objects
    """
    layers = [Image.open(io.BytesIO(l)) for l in renderer.render_layers(example, short_side_size=short_side_size)]
    alphas = np.array([np.array(layer.getchannel("A")) for layer in layers])
    layer_types = renderer.get_layer_types(example)
    layer_is_transparent = renderer.get_is_transparent(example)
    remaining_indices = list(range(0, len(alphas)))

    # Filter out unwanted layers
    for i, (layer_type, is_transparent) in enumerate(zip(layer_types, layer_is_transparent)):
        if (layer_type == "TextElement" and exclude_text) or (is_transparent and exclude_transparent):
            if i in remaining_indices:
                remaining_indices.remove(i)
        if alphas[i].min() == 255 and i != 0:  # Opaque layer other than background
            logger.warning(
                f"A foreground layer covers the whole canvas. Removing the hidden layers. (example id: {example['id']})"
            )
            remaining_indices = remaining_indices[i:]
    if len(remaining_indices) < 2:
        logger.warning(f"{len(remaining_indices)} layers are left after filtering. (example id: {example['id']})")
        return [], [], []

    top_layers_indices_list_rel = group_top_layers_indices(alphas[remaining_indices])
    # Map relative indices to absolute indices
    top_layers_indices_list = [[remaining_indices[i] for i in rel_inds] for rel_inds in top_layers_indices_list_rel]

    pairs: list[tuple[np.ndarray, Image.Image]] = []
    inpainted_pairs: list[tuple[np.ndarray, Image.Image]] = []

    if alphas[top_layers_indices_list[0]].min() == 0:
        logger.warning(
            f"Background layer is not covering the whole canvas. Adding white background. (example id: {example['id']})"
        )
        add_white_bg = True
    else:
        add_white_bg = False

    # Create pairs of matting target and input image
    for i in range(0, len(top_layers_indices_list)):  # Back to front
        if i == 0 and not add_white_bg:  # Background
            top_layers_mask = np.zeros_like(alphas[0])
            shrink_kernel = 0
        else:
            top_layers_mask = alphas[top_layers_indices_list[i]].max(axis=0)
            shrink_kernel = 7
        # Flattening indices
        backdrop_indices = [idx for group in top_layers_indices_list[: i + 1] for idx in group]
        backdrop = Image.open(
            io.BytesIO(renderer.render(example, short_side_size=short_side_size, layer_indices=backdrop_indices))
        )
        pairs.append((top_layers_mask, backdrop))

        if inpaint_model is not None and (i + 1 < len(top_layers_indices_list)):
            further_top_layers_indices = top_layers_indices_list[i + 1]
            _inpaint_masks = []  # Make inpainting mask
            # Check if each further top layer interferes with the current top layers
            for j in further_top_layers_indices:
                further_top_layer_mask = alphas[j]
                overlap = (shrink_mask(top_layers_mask, kernel_size=shrink_kernel) & further_top_layer_mask).sum()
                if overlap == 0 or overlap == (further_top_layer_mask > 0).sum():
                    _inpaint_masks.append(further_top_layer_mask)
            if len(_inpaint_masks) > 0:  # Inpainting
                inpaint_mask = expand_mask(np.max(_inpaint_masks, axis=0) > 0)
                backdrop_inpaint = inpaint_model.infer(np.array(backdrop.convert("RGB")), hard_mask=inpaint_mask)
                inpainted_pairs.append((top_layers_mask, Image.fromarray(backdrop_inpaint)))

    pairs = pairs[::-1]  # Reverse to have front-to-back order for saving
    inpainted_pairs = inpainted_pairs[::-1]

    if add_white_bg:
        white_bg = Image.new("RGBA", alphas[0].shape[::-1], (255, 255, 255, 255))
        pairs.append((np.zeros_like(alphas[0]), white_bg))
        if inpaint_model is not None:
            top_layers_mask = alphas[top_layers_indices_list[0]].max(axis=0)
            inpaint_mask = expand_mask(top_layers_mask > 0)
            bg_inpaint = inpaint_model.infer(np.array(white_bg.convert("RGB")), hard_mask=inpaint_mask)
            inpainted_pairs.append((np.zeros_like(alphas[0]), Image.fromarray(bg_inpaint)))

    layers_pil: list[Image.Image] = [layers[i] for i in remaining_indices]
    if add_white_bg:
        layers_pil = [Image.new("RGBA", layers[0].size, (255, 255, 255, 255))] + layers_pil

    return pairs, inpainted_pairs, layers_pil
