"""Layer organization and element extraction."""

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from layerd.models.helpers import divide_mask_to_connected_components
from layerd.types import BoundingBox, Element
from layerd.utils import apply_mask, assign_masks_to_groups, crop_image_with_bbox, mask_to_bbox

if TYPE_CHECKING:
    # Import for type checking only to avoid circular dependencies
    from typing import TypedDict

    class Polygon(TypedDict):
        """Polygon vertex for OCR results."""

        x: int
        y: int


class LayerOrganizer:
    """Organizer for LayerD decomposition results.

    Processes raw RGBA layers into structured elements with metadata.
    Supports optional OCR integration for text detection.

    Args:
        overlap_threshold: Minimum overlap ratio for OCR-CC matching (default: 0.9)
        labeler: Element classification strategy (optional, from layerd.classification)

    Example:
        >>> from layerd import LayerD
        >>> from layerd.postprocess import LayerOrganizer
        >>> from PIL import Image
        >>>
        >>> # Decompose image
        >>> layerd = LayerD()
        >>> image = Image.open("design.png")
        >>> layers = layerd.decompose(image)
        >>>
        >>> # Organize into elements (without OCR)
        >>> organizer = LayerOrganizer()
        >>> elements = organizer.organize(layers)
        >>>
        >>> # Access organized elements
        >>> for elem in elements:
        ...     print(f"Element {elem['id']}: type={elem['type']}, bbox={elem['box']}")
    """

    def __init__(
        self,
        overlap_threshold: float = 0.9,
        labeler: Any | None = None,
    ) -> None:
        """Initialize layer organizer.

        Args:
            overlap_threshold: Minimum overlap ratio for OCR-CC matching
            labeler: Element classification strategy (None to skip classification)
                    Should have a classify(elements: list[Element]) -> list[Element] method
        """
        self.overlap_threshold = overlap_threshold
        self.labeler = labeler

    def organize(
        self,
        layers: list[Image.Image],
        ocr_result: dict[str, Any] | None = None,
    ) -> list[Element]:
        """Organize LayerD decomposition results into structured elements.

        Args:
            layers: List of RGBA PIL Images from LayerD decomposition
            ocr_result: Optional OCR result with format:
                       {"image_size": (width, height),
                        "blocks": [{"text": str, "bbox": BoundingBox, ...}, ...]}
                       If None, all elements are classified as "image" type.

        Returns:
            List of organized elements with cropped images and bounding boxes

        Example:
            >>> # Without OCR
            >>> organizer = LayerOrganizer()
            >>> elements = organizer.organize(layers)
            >>>
            >>> # With OCR (requires layerd-internal or custom OCR)
            >>> ocr_result = {"image_size": (800, 600), "blocks": [...]}
            >>> elements = organizer.organize(layers, ocr_result)
        """
        if not layers:
            raise ValueError("At least one layer is required")

        # Determine canvas size
        if ocr_result is not None:
            canvas_size = ocr_result["image_size"]
        else:
            # Use first layer dimensions as canvas size
            canvas_size = layers[0].size

        # Pre-compute OCR block masks if OCR result is provided
        block_masks: list[np.ndarray] = []
        if ocr_result is not None and "blocks" in ocr_result:
            ocr_blocks = ocr_result["blocks"]
            for block in ocr_blocks:
                # Use polygon if available, otherwise use rectangular bbox
                if "polygon" in block and block["polygon"]:
                    block_mask = self._polygon_to_mask(block["polygon"], canvas_size)
                else:
                    block_mask = self._bbox_to_mask(block["bbox"], canvas_size)
                block_masks.append(block_mask)

        elements: list[Element] = []
        element_id = 0

        for layer in layers:
            # Get layer alpha mask
            layer_np = np.array(layer)
            alpha = layer_np[:, :, 3] if layer_np.shape[2] == 4 else np.ones(layer_np.shape[:2], dtype=np.uint8) * 255
            layer_mask = alpha > 0

            # Divide into connected components
            cc_masks = divide_mask_to_connected_components(layer_mask)

            if block_masks:
                # Assign connected components to OCR blocks
                block_combined_masks, unassigned_ccs = assign_masks_to_groups(
                    cc_masks, block_masks, self.overlap_threshold
                )

                # Create text elements from matched OCR blocks
                for combined_mask in block_combined_masks:
                    if np.any(combined_mask > 0):
                        bbox = mask_to_bbox(combined_mask)
                        masked = apply_mask(layer, combined_mask)
                        cropped = crop_image_with_bbox(masked, bbox)
                        elements.append(Element(id=element_id, type="text", image=cropped, box=bbox))
                        element_id += 1

                # Create image elements from unassigned CCs
                for cc_mask in unassigned_ccs:
                    bbox = mask_to_bbox(cc_mask)
                    masked = apply_mask(layer, cc_mask)
                    cropped = crop_image_with_bbox(masked, bbox)
                    elements.append(Element(id=element_id, type="image", image=cropped, box=bbox))
                    element_id += 1
            else:
                # No OCR - all connected components become "image" elements
                for cc_mask in cc_masks:
                    bbox = mask_to_bbox(cc_mask)
                    masked = apply_mask(layer, cc_mask)
                    cropped = crop_image_with_bbox(masked, bbox)
                    elements.append(Element(id=element_id, type="image", image=cropped, box=bbox))
                    element_id += 1

        # Apply classification if labeler is provided
        if self.labeler is not None:
            elements = self.labeler.classify(elements)

        return elements

    def __call__(
        self,
        layers: list[Image.Image],
        ocr_result: dict[str, Any] | None = None,
    ) -> list[Element]:
        """Convenience method for organize().

        Args:
            layers: List of RGBA PIL Images
            ocr_result: Optional OCR result

        Returns:
            List of organized elements
        """
        return self.organize(layers, ocr_result)

    def _bbox_to_mask(self, bbox: BoundingBox, canvas_size: tuple[int, int]) -> np.ndarray:
        """Convert rectangular bounding box to binary mask.

        Args:
            bbox: Bounding box with x_min, y_min, x_max, y_max
            canvas_size: (width, height) of the canvas

        Returns:
            Binary mask as uint8 numpy array (H, W)
        """
        width, height = canvas_size
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[bbox["y_min"] : bbox["y_max"], bbox["x_min"] : bbox["x_max"]] = 255
        return mask

    def _polygon_to_mask(self, polygon: list[dict[str, int]], canvas_size: tuple[int, int]) -> np.ndarray:
        """Convert polygon vertices to binary mask.

        Args:
            polygon: List of vertices with x and y coordinates
                    Each vertex is a dict with 'x' and 'y' keys
            canvas_size: (width, height) of the canvas

        Returns:
            Binary mask as uint8 numpy array (H, W)
        """
        width, height = canvas_size
        mask = np.zeros((height, width), dtype=np.uint8)

        # Convert vertices to numpy array
        points = np.array([[v["x"], v["y"]] for v in polygon], dtype=np.int32)

        # Fill polygon
        cv2.fillPoly(mask, [points], (255,))

        return mask
