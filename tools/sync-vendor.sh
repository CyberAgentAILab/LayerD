#!/bin/bash
# tools/sync-vendor.sh
# Manually sync vendored dependencies from vendor/ to src/layerd/_vendor/

set -e

echo "Syncing vendored dependencies..."

exit_code=0

# Sync simple-lama-inpainting
if [ -d "vendor/simple-lama-inpainting/simple_lama_inpainting" ]; then
    echo "  - Syncing simple-lama-inpainting..."
    rm -rf src/layerd/_vendor/simple_lama_inpainting
    cp -r vendor/simple-lama-inpainting/simple_lama_inpainting/ \
        src/layerd/_vendor/simple_lama_inpainting/
else
    echo "  ⚠️  vendor/simple-lama-inpainting not found"
    exit_code=1
fi

# Sync cr-renderer
if [ -d "vendor/cr-renderer/src/cr_renderer" ]; then
    echo "  - Syncing cr-renderer..."
    rm -rf src/layerd/_vendor/cr_renderer
    cp -r vendor/cr-renderer/src/cr_renderer/ \
        src/layerd/_vendor/cr_renderer/
else
    echo "  ⚠️  vendor/cr-renderer not found"
    exit_code=1
fi

if [ $exit_code -ne 0 ]; then
    echo "✗ Sync incomplete - some directories were missing"
    exit $exit_code
fi

echo "✓ Sync complete!"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff src/layerd/_vendor/"
echo "  2. Run tests: uv run pytest"
echo "  3. Stage changes: git add src/layerd/_vendor/"
echo "  4. Commit: git commit -m 'sync: update vendored dependencies'"
