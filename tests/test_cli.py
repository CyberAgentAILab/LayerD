"""Tests for the LayerD CLI."""

import os
import sys
from pathlib import Path

import pytest

from layerd.cli import main, parse_args, run_decompose

# Test with sample image
TEST_IMAGE_PATH = Path(__file__).parent.parent / "data" / "test_image_2.png"


def test_parse_args_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test argument parsing with minimal required arguments."""
    test_args = [
        "layerd",
        "--input",
        "test.png",
        "--output-dir",
        "/tmp/output",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    args = parse_args()

    assert args.input == "test.png"
    assert args.output_dir == "/tmp/output"
    assert args.device == "cpu"  # default
    assert args.max_iterations == 3  # default
    assert args.matting_hf_card == "cyberagent/layerd-birefnet"  # default
    assert args.matting_process_size is None  # default
    assert args.log_level == "INFO"  # default


def test_parse_args_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test argument parsing with all arguments specified."""
    test_args = [
        "layerd",
        "--input",
        "test.png",
        "--output-dir",
        "/tmp/output",
        "--device",
        "cuda",
        "--max-iterations",
        "5",
        "--matting-hf-card",
        "custom/model",
        "--matting-process-size",
        "512",
        "512",
        "--log-level",
        "DEBUG",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    args = parse_args()

    assert args.input == "test.png"
    assert args.output_dir == "/tmp/output"
    assert args.device == "cuda"
    assert args.max_iterations == 5
    assert args.matting_hf_card == "custom/model"
    assert args.matting_process_size == [512, 512]
    assert args.log_level == "DEBUG"


def test_run_decompose_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that run_decompose raises ValueError for missing input file."""
    test_args = [
        "layerd",
        "--input",
        str(tmp_path / "nonexistent.png"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    args = parse_args()

    with pytest.raises(ValueError, match="Input file does not exist"):
        run_decompose(args)


def test_run_decompose_invalid_format(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that run_decompose raises ValueError for invalid file format."""
    # Create a text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    test_args = [
        "layerd",
        "--input",
        str(test_file),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    args = parse_args()

    with pytest.raises(ValueError, match="Invalid file format"):
        run_decompose(args)


def test_run_decompose_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test successful decomposition with run_decompose."""
    output_dir = tmp_path / "output"

    test_args = [
        "layerd",
        "--input",
        str(TEST_IMAGE_PATH),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--max-iterations",
        "2",  # Use fewer iterations for faster test
        "--log-level",
        "WARNING",  # Reduce log output
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    args = parse_args()
    run_decompose(args)

    # Check that output directory was created
    expected_output_dir = output_dir / "test_image_2"
    assert expected_output_dir.exists()
    assert expected_output_dir.is_dir()

    # Check that layer files were created
    layer_files = sorted(expected_output_dir.glob("*.png"))
    assert len(layer_files) >= 1  # At least one layer (background)

    # Check layer file naming
    for i, layer_file in enumerate(layer_files):
        expected_name = f"{i:04d}.png"
        assert layer_file.name == expected_name


def test_main_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test the main() entry point with valid input."""
    output_dir = tmp_path / "output"

    test_args = [
        "layerd",
        "--input",
        str(TEST_IMAGE_PATH),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--max-iterations",
        "2",
        "--log-level",
        "WARNING",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # main() should not raise an exception
    main()

    # Verify output was created
    expected_output_dir = output_dir / "test_image_2"
    assert expected_output_dir.exists()
    layer_files = list(expected_output_dir.glob("*.png"))
    assert len(layer_files) >= 1


def test_main_error_handling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that main() handles errors gracefully with SystemExit."""
    test_args = [
        "layerd",
        "--input",
        str(tmp_path / "nonexistent.png"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # main() should raise SystemExit(1) for ValueError
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1


def test_output_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that output files follow the expected structure."""
    output_dir = tmp_path / "output"

    test_args = [
        "layerd",
        "--input",
        str(TEST_IMAGE_PATH),
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--max-iterations",
        "3",
        "--log-level",
        "WARNING",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    args = parse_args()
    run_decompose(args)

    # Check output structure: output_dir/filename/0000.png, 0001.png, ...
    expected_subdir = output_dir / "test_image_2"
    assert expected_subdir.exists()

    layer_files = sorted(expected_subdir.glob("*.png"))
    assert len(layer_files) >= 1

    # Verify zero-padded naming (0000.png, 0001.png, ...)
    for i, layer_file in enumerate(layer_files):
        assert layer_file.name == f"{i:04d}.png"
        # Verify files are not empty
        assert layer_file.stat().st_size > 0
