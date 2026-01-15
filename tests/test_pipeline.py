"""Unit tests for LayerDPipeline with mocked components."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from layerd.classification import EntropyLabeler
from layerd.pipeline import LayerDPipeline, PipelineResult
from layerd.types import BoundingBox, Element


def create_dummy_image(width: int = 512, height: int = 512) -> Image.Image:
    """Create a dummy RGB image for testing."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def create_dummy_layer(width: int = 512, height: int = 512) -> Image.Image:
    """Create a dummy RGBA layer for testing."""
    return Image.new("RGBA", (width, height), color=(255, 100, 100, 255))


def create_dummy_element() -> Element:
    """Create a dummy element."""
    return Element(
        id=0,
        type="image",
        image=create_dummy_layer(),
        box=BoundingBox(x_min=50, y_min=50, x_max=200, y_max=100),
    )


class TestPipelineInitialization:
    """Test LayerDPipeline initialization."""

    def test_init_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        pipeline = LayerDPipeline()
        assert pipeline.device == "cpu"
        assert pipeline.overlap_threshold == 0.9
        assert isinstance(pipeline.labeler, EntropyLabeler)
        assert pipeline.labeler.threshold == 5.0

    def test_init_custom_layerd_parameters(self) -> None:
        """Test initialization with custom LayerD parameters."""
        pipeline = LayerDPipeline(
            matting_process_size=(512, 512),
            use_unblend=False,
            bg_refine=False,
            fg_refine=False,
        )
        assert pipeline.layerd is not None

    def test_init_custom_organizer_parameters(self) -> None:
        """Test initialization with custom LayerOrganizer parameters."""
        pipeline = LayerDPipeline(
            overlap_threshold=0.7,
            labeler=None,  # Disable classification
        )
        assert pipeline.overlap_threshold == 0.7
        assert pipeline.labeler is None

    def test_init_custom_labeler(self) -> None:
        """Test initialization with custom labeler threshold."""
        pipeline = LayerDPipeline(labeler_threshold=3.0)
        assert isinstance(pipeline.labeler, EntropyLabeler)
        assert pipeline.labeler.threshold == 3.0


class TestPipelineExecution:
    """Test pipeline execution."""

    @patch("layerd.pipeline.LayerD")
    @patch("layerd.pipeline.LayerOrganizer")
    def test_pipeline_basic_execution(
        self,
        mock_organizer_cls: MagicMock,
        mock_layerd_cls: MagicMock,
    ) -> None:
        """Test complete pipeline execution."""
        # Setup mocks
        mock_layerd = MagicMock()
        mock_layerd.decompose.return_value = [create_dummy_layer(), create_dummy_layer()]
        mock_layerd_cls.return_value = mock_layerd

        mock_organizer = MagicMock()
        mock_organizer.organize.return_value = [create_dummy_element()]
        mock_organizer_cls.return_value = mock_organizer

        # Create pipeline and run
        pipeline = LayerDPipeline()
        image = create_dummy_image()
        result = pipeline(image, max_iterations=3)

        # Verify result structure
        assert isinstance(result, PipelineResult)
        assert len(result.layers) == 2
        assert len(result.elements) == 1
        assert result.ocr_result is None  # Phase 2: no OCR
        assert result.canvas_size == (512, 512)

        # Verify method calls
        mock_layerd.decompose.assert_called_once_with(image, max_iterations=3)
        mock_organizer.organize.assert_called_once()

    @patch("layerd.pipeline.LayerD")
    @patch("layerd.pipeline.LayerOrganizer")
    def test_pipeline_layers_are_rgba(
        self,
        mock_organizer_cls: MagicMock,
        mock_layerd_cls: MagicMock,
    ) -> None:
        """Test that pipeline returns RGBA layers."""
        # Setup mocks
        mock_layerd = MagicMock()
        layer1 = create_dummy_layer()
        layer2 = create_dummy_layer()
        mock_layerd.decompose.return_value = [layer1, layer2]
        mock_layerd_cls.return_value = mock_layerd

        mock_organizer = MagicMock()
        mock_organizer.organize.return_value = [create_dummy_element()]
        mock_organizer_cls.return_value = mock_organizer

        # Run pipeline
        pipeline = LayerDPipeline()
        result = pipeline(create_dummy_image())

        # Verify layers
        assert all(layer.mode == "RGBA" for layer in result.layers)


class TestDeviceSwitching:
    """Test device switching behavior."""

    @patch("layerd.pipeline.LayerD")
    def test_device_switching(self, mock_layerd_cls: MagicMock) -> None:
        """Test device switching."""
        mock_layerd = MagicMock()
        mock_layerd.to.return_value = mock_layerd
        mock_layerd_cls.return_value = mock_layerd

        pipeline = LayerDPipeline(device="cpu")
        assert pipeline.device == "cpu"

        # Switch device
        result = pipeline.to("cuda")
        assert result is pipeline  # Returns self
        assert pipeline.device == "cuda"
        mock_layerd.to.assert_called_once_with("cuda")


class TestErrorHandling:
    """Test error handling in pipeline."""

    @patch("layerd.pipeline.LayerD")
    def test_layerd_decomposition_error(self, mock_layerd_cls: MagicMock) -> None:
        """Test error handling when LayerD decomposition fails."""
        mock_layerd = MagicMock()
        mock_layerd.decompose.side_effect = RuntimeError("Decomposition failed")
        mock_layerd_cls.return_value = mock_layerd

        pipeline = LayerDPipeline()

        with pytest.raises(RuntimeError, match="LayerD decomposition failed"):
            pipeline(create_dummy_image())

    @patch("layerd.pipeline.LayerD")
    @patch("layerd.pipeline.LayerOrganizer")
    def test_organizer_error(
        self,
        mock_organizer_cls: MagicMock,
        mock_layerd_cls: MagicMock,
    ) -> None:
        """Test error handling when LayerOrganizer fails."""
        mock_layerd = MagicMock()
        mock_layerd.decompose.return_value = [create_dummy_layer()]
        mock_layerd_cls.return_value = mock_layerd

        mock_organizer = MagicMock()
        mock_organizer.organize.side_effect = RuntimeError("Organizer failed")
        mock_organizer_cls.return_value = mock_organizer

        pipeline = LayerDPipeline()

        with pytest.raises(RuntimeError, match="Layer organization failed"):
            pipeline(create_dummy_image())


class TestPipelineResult:
    """Test PipelineResult model."""

    def test_pipeline_result_structure(self) -> None:
        """Test PipelineResult structure and types."""
        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        assert len(result.elements) == 1
        assert len(result.layers) == 1
        assert result.ocr_result is None
        assert result.canvas_size == (512, 512)

    @patch("layerd.pipeline.SVGBuilder")
    def test_to_svg_base64(self, mock_svg_builder: MagicMock) -> None:
        """Test SVG export with base64 mode."""
        mock_builder = MagicMock()
        mock_builder.return_value = "<svg>test</svg>"
        mock_svg_builder.return_value = mock_builder

        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        svg = result.to_svg()
        assert svg == "<svg>test</svg>"
        mock_svg_builder.assert_called_once_with(image_mode="base64", image_dir=None)

    @patch("layerd.pipeline.SVGBuilder")
    def test_to_svg_external(self, mock_svg_builder: MagicMock) -> None:
        """Test SVG export with external images."""
        mock_builder = MagicMock()
        mock_builder.return_value = "<svg>test</svg>"
        mock_svg_builder.return_value = mock_builder

        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        svg = result.to_svg(image_mode="external", image_dir="./images")
        assert svg == "<svg>test</svg>"
        mock_svg_builder.assert_called_once_with(image_mode="external", image_dir="./images")

    def test_to_svg_external_without_dir_raises(self) -> None:
        """Test that external mode without image_dir raises ValueError."""
        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        with pytest.raises(ValueError, match="image_dir required"):
            result.to_svg(image_mode="external")

    @patch("layerd.pipeline.build_exporter")
    def test_to_psd(self, mock_build_exporter: MagicMock) -> None:
        """Test PSD export."""
        mock_builder = MagicMock()
        mock_builder.return_value = b"PSD data"
        mock_build_exporter.return_value = mock_builder

        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        psd_bytes = result.to_psd()
        assert psd_bytes == b"PSD data"
        mock_build_exporter.assert_called_once_with("psd", compression="rle", color_depth=8)

    @patch("layerd.pipeline.build_exporter")
    def test_to_psd_custom_options(self, mock_build_exporter: MagicMock) -> None:
        """Test PSD export with custom options."""
        mock_builder = MagicMock()
        mock_builder.return_value = b"PSD data"
        mock_build_exporter.return_value = mock_builder

        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        psd_bytes = result.to_psd(compression="zip", color_depth=16)
        assert psd_bytes == b"PSD data"
        mock_build_exporter.assert_called_once_with("psd", compression="zip", color_depth=16)

    @patch("layerd.pipeline.Path.mkdir")
    @patch("builtins.open", create=True)
    @patch("layerd.pipeline.SVGBuilder")
    def test_save_svg_autodetect(
        self,
        mock_svg_builder: MagicMock,
        mock_open: MagicMock,
        mock_mkdir: MagicMock,
    ) -> None:
        """Test save with auto-detected SVG format."""
        mock_builder = MagicMock()
        mock_builder.return_value = "<svg>test</svg>"
        mock_svg_builder.return_value = mock_builder

        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        result.save("output.svg")
        mock_open.assert_called_once_with("output.svg", "w")

    @patch("layerd.pipeline.Path.mkdir")
    @patch("builtins.open", create=True)
    @patch("layerd.pipeline.build_exporter")
    def test_save_psd_autodetect(
        self,
        mock_build_exporter: MagicMock,
        mock_open: MagicMock,
        mock_mkdir: MagicMock,
    ) -> None:
        """Test save with auto-detected PSD format."""
        mock_builder = MagicMock()
        mock_builder.return_value = b"PSD data"
        mock_build_exporter.return_value = mock_builder

        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        result.save("output.psd")
        mock_open.assert_called_once_with("output.psd", "wb")

    def test_save_unknown_extension_raises(self) -> None:
        """Test that unknown extension raises ValueError."""
        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        with pytest.raises(ValueError, match="Cannot determine format"):
            result.save("output.txt")

    @patch("layerd.pipeline.Path.mkdir")
    @patch("builtins.open", create=True)
    @patch("layerd.pipeline.SVGBuilder")
    def test_save_explicit_format(
        self,
        mock_svg_builder: MagicMock,
        mock_open: MagicMock,
        mock_mkdir: MagicMock,
    ) -> None:
        """Test save with explicit format parameter."""
        mock_builder = MagicMock()
        mock_builder.return_value = "<svg>test</svg>"
        mock_svg_builder.return_value = mock_builder

        result = PipelineResult(
            elements=[create_dummy_element()],
            layers=[create_dummy_layer()],
            ocr_result=None,
            canvas_size=(512, 512),
        )

        result.save("output", format="svg")
        mock_open.assert_called_once_with("output", "w")
