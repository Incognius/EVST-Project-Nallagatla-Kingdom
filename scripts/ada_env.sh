#!/bin/bash
export UV_CACHE_DIR=/scratch/$USER/uvc UV_PYTHON_INSTALL_DIR=$HOME/.uv_python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=900
mkdir -p /scratch/$USER
U=~/.local/bin/uv
P=~/venvs/evst/bin/python
set -e
$U venv --allow-existing --python 3.12 ~/venvs/evst
# torch + CUDA libs from the PyTorch CDN (fast from ada); everything else prefers the local wheel cache
$U pip install --python $P torch --index-url https://download.pytorch.org/whl/cu121
$U pip install --python $P --find-links ~/wheels numpy pandas scipy scikit-learn joblib matplotlib
$U pip install --python $P --find-links ~/wheels rasterio pyproj lightgbm
cd ~/evst && $U pip install --python $P --no-deps -e .
$P -c "import torch, rasterio, lightgbm, pandas; print('ENV OK', torch.__version__, torch.cuda.is_available(), rasterio.__version__)"
