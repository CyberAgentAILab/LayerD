import numpy as np
from PIL import Image

from layerd.models.inpaint import build_inpaint
from layerd.models.matting import build_matting

from .helpers import (
    estimate_fg_alpha,
    estimate_fg_color,
    expand_mask,
    find_flat_color_region_ccs,
    refine_background,
    shrink_mask_ratio,
)


class LayerD:
    def __init__(
        self,
        matting_hf_card: str = "cyberagent/layerd-birefnet",
        matting_process_size: tuple[int, int] | None = None,
        matting_weight_path: str | None = None,
        use_unblend: bool = True,
        bg_refine: bool = True,
        fg_refine: bool = True,
        fg_refine_num_colors: int = 2,
        bg_refine_num_colors: int = 10,
        kernel_scale: float = 0.015,
        device: str = "cpu",
    ) -> None:
        """Initialize LayerD model for image decomposition.

        Args:
            matting_hf_card: Hugging Face model card for the matting model.
            matting_process_size: Optional size (width, height) to resize images for matting processing.
            matting_weight_path: Optional path to local model weights. Overrides hugging face model if provided.
            use_unblend: Whether to use unblending technique for foreground color estimation.
            bg_refine: Whether to refine background with palette-based color assignment.
            fg_refine: Whether to refine foreground alpha and colors using flat color regions.
            fg_refine_num_colors: Number of colors for foreground refinement.
            bg_refine_num_colors: Number of colors for background refinement.
            kernel_scale: Scale factor to determine kernel size for mask expansion based on image dimensions.
            device: Device to run models on ("cpu" or "cuda").
        """

        self.matting_model = build_matting(
            "birefnet",
            hf_card=matting_hf_card,
            process_image_size=matting_process_size,
            weight_path=matting_weight_path,
        )
        self.inpaint_model = build_inpaint("lama")
        self.use_unblend = use_unblend
        self.bg_refine = bg_refine
        self.fg_refine = fg_refine

        # Parameters for refinement
        self.fg_refine_num_colors = fg_refine_num_colors
        self.bg_refine_num_colors = bg_refine_num_colors
        self._kernel_scale = kernel_scale
        self._th_alpha = 0.005  # threshold for hard alpha mask
        self._unblend_alpha_clip = [0, 0.95]  # clipping range for unblending
        self._palette_percentile = 0.99  # percentile for palette color selection in both fg and bg refinement
        self._bg_refine_n_outer_ratio = 0.2  # ratio for outer region to determine bg flatness
        self._fg_refine_n_inner_ratio = 0.1  # ratio for inner region to be refined
        self.to(device)

    def _calc_kernel_size(self, image: Image.Image) -> tuple[int, int]:
        kernel_size = (round(image.height * self._kernel_scale), round(image.width * self._kernel_scale))
        return kernel_size

    def _decompose_step(self, image: Image.Image) -> tuple[Image.Image | None, Image.Image]:
        image_rgb = np.array(image.convert("RGB"))
        kernel_size = self._calc_kernel_size(image)

        alpha = self.matting_model(image)
        hard_mask = alpha > self._th_alpha
        if hard_mask.sum() == 0:  # No content
            return None, image
        if np.mean(hard_mask) > 0.99:  # Full content
            return None, image

        if self.fg_refine:
            color_masks, colors, ccs = find_flat_color_region_ccs(
                image_rgb, hard_mask, max_num_colors=self.fg_refine_num_colors, percentile=self._palette_percentile
            )
            # Shrink connected components to be refined
            shrinked_ccs = [
                ccs[i] if len(colors[i]) == 0 else shrink_mask_ratio(ccs[i], self._fg_refine_n_inner_ratio)
                for i in range(len(ccs))
            ]
            inpaint_mask = expand_mask(np.any(shrinked_ccs + sum(color_masks, []), axis=0), kernel_size)
        else:
            inpaint_mask = expand_mask(hard_mask, kernel_size=kernel_size)

        bg = self.inpaint_model(image_rgb, inpaint_mask)

        if self.bg_refine:
            bg = refine_background(
                bg, inpaint_mask, max_num_colors=self.bg_refine_num_colors, n_outer_ratio=self._bg_refine_n_outer_ratio
            )

        if self.use_unblend:
            fg_rgb = estimate_fg_color(image_rgb, bg, alpha, self._unblend_alpha_clip)
        else:
            fg_rgb = image_rgb.copy()

        if self.fg_refine:
            for colors_cc, color_masks_cc, cc in zip(colors, color_masks, ccs):
                _refined_alpha = np.zeros_like(alpha)
                _refined_color = np.zeros_like(fg_rgb)
                _nonzero_mask_counts = np.zeros_like(alpha)
                for color, color_mask in zip(colors_cc, color_masks_cc):
                    color_mask_expanded = expand_mask(color_mask, kernel_size)
                    _refined_alpha_color = estimate_fg_alpha(color_mask_expanded, color, bg, image_rgb)
                    if _refined_alpha_color is not None:
                        _refined_alpha = np.maximum(_refined_alpha, _refined_alpha_color)
                        _refined_color[_refined_alpha_color > 0] = color
                        _nonzero_mask_counts += (_refined_alpha_color > 0).astype(int)
                color_boundary_mask = _nonzero_mask_counts > 1
                if _refined_alpha.sum() > 0:
                    inner_cc = (~shrink_mask_ratio(cc, self._fg_refine_n_inner_ratio)) & cc
                    target_mask = ((alpha == 0) | inner_cc) & (~color_boundary_mask)
                    alpha[target_mask] = np.maximum(alpha[target_mask], _refined_alpha[target_mask])
                    fg_rgb[target_mask & (_refined_alpha > 0)] = _refined_color[target_mask & (_refined_alpha > 0)]

        background = Image.fromarray(bg)
        foreground = Image.fromarray(np.dstack([fg_rgb, np.array(alpha * 255, dtype=np.uint8)])).convert("RGBA")

        return foreground, background

    def decompose(self, image: Image.Image, max_iterations: int = 3) -> list[Image.Image]:
        """Decompose an image into layers of foregrounds and backgrounds.
        Args:
            image: Input PIL Image to decompose.
            max_iterations: Maximum number of decomposition iterations.
        Returns:
            List of PIL Images representing the layers, starting with the final background.
        """

        bg_list = []
        fg_list = []
        current_bg = image.convert("RGB")

        for _ in range(max_iterations):
            fg, new_bg = self._decompose_step(current_bg)
            if fg is None:
                break

            bg_list.append(new_bg)
            fg_list.append(fg)
            current_bg = new_bg

        if len(fg_list) == 0:
            return [image.convert("RGBA")]

        final_bg = bg_list[-1].convert("RGBA")
        return [final_bg] + fg_list[::-1]

    def to(self, device: str) -> "LayerD":
        self.matting_model.to(device)
        self.inpaint_model.to(device)
        return self
