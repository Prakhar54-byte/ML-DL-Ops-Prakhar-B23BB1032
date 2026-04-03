#!/usr/bin/env bash
# fix_inside_container.sh
# Run this INSIDE the running Docker container to fix PyTorch CUDA version
# Usage: bash fix_inside_container.sh

set -e

echo "=== Current PyTorch version ==="
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "=== Installing PyTorch 2.5.1 for CUDA 12.4 ==="
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --upgrade --quiet

echo ""
echo "=== Verifying fix ==="
python -c "
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
else:
    print('WARNING: Still no CUDA - will train on CPU')
"
echo ""
echo "=== PyTorch fix done. Now run: bash run_all.sh ==="
