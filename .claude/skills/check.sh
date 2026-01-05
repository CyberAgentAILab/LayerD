#!/bin/bash
# Quality checks skill for LayerD
# Usage: /check [mypy|ruff|format|test|all]

set -e

COMMAND=${1:-all}

run_mypy() {
    echo "Running type checking..."
    uv run mypy src/ tests/ \
        vendor/simple-lama-inpainting/simple_lama_inpainting/ \
        vendor/cr-renderer/src/cr_renderer/
}

run_ruff_check() {
    echo "Running linting..."
    uv run ruff check src/ tests/ vendor/
}

run_ruff_format() {
    echo "Running formatter check..."
    uv run ruff format src/ tests/ vendor/ --check
}

run_tests() {
    echo "Running tests..."
    uv run pytest
}

run_all() {
    run_mypy
    echo ""
    run_ruff_check
    echo ""
    run_ruff_format
    echo ""
    run_tests
}

case "$COMMAND" in
    mypy)
        run_mypy
        ;;
    ruff)
        run_ruff_check
        ;;
    format)
        run_ruff_format
        ;;
    test)
        run_tests
        ;;
    all)
        run_all
        ;;
    *)
        echo "Usage: /check [mypy|ruff|format|test|all]"
        echo ""
        echo "Commands:"
        echo "  mypy   - Run type checking with mypy"
        echo "  ruff   - Run linting with ruff"
        echo "  format - Check code formatting"
        echo "  test   - Run pytest"
        echo "  all    - Run all checks (default)"
        exit 1
        ;;
esac
