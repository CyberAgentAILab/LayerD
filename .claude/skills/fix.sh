#!/bin/bash
# Auto-fix skill for LayerD
# Usage: /fix [ruff|format|all]

set -e

COMMAND=${1:-all}

run_ruff_fix() {
    echo "Auto-fixing linting issues..."
    uv run ruff check src/ tests/ vendor/ --fix
}

run_format() {
    echo "Auto-formatting code..."
    uv run ruff format src/ tests/ vendor/
}

run_all() {
    run_ruff_fix
    echo ""
    run_format
}

case "$COMMAND" in
    ruff)
        run_ruff_fix
        ;;
    format)
        run_format
        ;;
    all)
        run_all
        ;;
    *)
        echo "Usage: /fix [ruff|format|all]"
        echo ""
        echo "Commands:"
        echo "  ruff   - Auto-fix linting issues with ruff"
        echo "  format - Auto-format code with ruff"
        echo "  all    - Run both (default)"
        exit 1
        ;;
esac
