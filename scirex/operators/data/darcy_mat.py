# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# For any clarifications or special considerations,
# please contact: contact@scirex.org

"""
Loader for the original FNO Darcy Flow MATLAB dataset.

Dataset source
--------------
Original files from Li et al. 2021 (arXiv:2010.08895):
    piececonst_r421_N1024_smooth1.mat   <- 1024 train samples at 421x421
    piececonst_r421_N1024_smooth2.mat   <- 1024 test  samples at 421x421

Download:
    pip install gdown
    python3 -c "
    import gdown, os
    os.makedirs('scirex/operators/data/darcy_original', exist_ok=True)
    gdown.download(
        'https://drive.google.com/uc?id=1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV',
        'scirex/operators/data/darcy_original/piececonst_r421_N1024_smooth1.mat')
    gdown.download(
        'https://drive.google.com/uc?id=1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf',
        'scirex/operators/data/darcy_original/piececonst_r421_N1024_smooth2.mat')
    "

IMPORTANT: File format
-----------------------
These are MATLAB v5 (.mat) files.
    USE   : scipy.io.loadmat      (correct)
    DO NOT: h5py                  (raises 'file signature not found')
    DO NOT: torch.load            (not a PyTorch file)

h5py only works for MATLAB v7.3 files saved with '-v7.3' flag.
These files were NOT saved that way.

Keys inside each file:
    'coeff' : (1024, 421, 421) float64  permeability a(x,y)
    'sol'   : (1024, 421, 421) float64  pressure     u(x,y)

Paper preprocessing (exact)
----------------------------
Subsample 421x421 to 85x85 using stride r=5:
    h = ((421-1)//5) + 1 = 85
    x = coeff[:n, ::5, ::5][:, :85, :85]
    y = sol  [:n, ::5, ::5][:, :85, :85]

FNO2D input: (n, 85, 85, 3) = [coeff, grid_x, grid_y]
FNO2D output:(n, 85, 85, 1) = [sol]

Paper result: n_train=1000, modes=12, width=32 -> Rel-L2 ~ 0.9%
"""

from typing import Tuple
import numpy as np


def load_darcy_mat(
    train_path: str,
    test_path:  str,
    n_train:    int = 1000,
    n_test:     int = 100,
    subsample_rate: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load FNO Darcy .mat files and return arrays for SciREX FNO2D.

    Args:
        train_path      : path to piececonst_r421_N1024_smooth1.mat
        test_path       : path to piececonst_r421_N1024_smooth2.mat
        n_train         : training samples to use (<=1024)
        n_test          : test samples to use (<=1024)
        subsample_rate  : stride when subsampling 421x421 grid.
                          r=5  -> 85x85  (paper default, ~0.9% Rel-L2)
                          r=3  -> 141x141 (higher res, slower)
                          r=10 -> 43x43  (fast, lower accuracy)

    Returns:
        x_train : (n_train, h, h, 3) float32  [coeff, grid_x, grid_y]
        y_train : (n_train, h, h, 1) float32  [sol]
        x_test  : (n_test,  h, h, 3) float32
        y_test  : (n_test,  h, h, 1) float32
        where h = ((421-1)//subsample_rate) + 1
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError(
            "scipy is required to load MATLAB v5 .mat files.\n"
            "  pip install scipy\n\n"
            "Do NOT use h5py — these are v5 format (not HDF5).\n"
            "h5py raises 'file signature not found' on these files."
        )

    r = subsample_rate
    h = int(((421 - 1) / r) + 1)   # 85 when r=5

    # Load and subsample
    print(f"Loading {train_path} ...")
    tr = loadmat(train_path)
    coeff_tr = tr['coeff'][:n_train, ::r, ::r][:, :h, :h]   # (n, h, h)
    sol_tr   = tr['sol']  [:n_train, ::r, ::r][:, :h, :h]

    print(f"Loading {test_path} ...")
    te = loadmat(test_path)
    coeff_te = te['coeff'][:n_test, ::r, ::r][:, :h, :h]
    sol_te   = te['sol']  [:n_test, ::r, ::r][:, :h, :h]

    # Normalised grid channels [0, 1]^2  (same as FNO paper)
    xs = np.linspace(0.0, 1.0, h, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    GX, GY = np.meshgrid(xs, ys, indexing='ij')   # (h, h)

    def _stack(coeff):
        n  = coeff.shape[0]
        c  = coeff.astype(np.float32)[:, :, :, None]
        gx = np.tile(GX[None, :, :, None], (n, 1, 1, 1))
        gy = np.tile(GY[None, :, :, None], (n, 1, 1, 1))
        return np.concatenate([c, gx, gy], axis=-1)   # (n,h,h,3)

    x_train = _stack(coeff_tr)
    x_test  = _stack(coeff_te)
    y_train = sol_tr.astype(np.float32)[:, :, :, None]   # (n,h,h,1)
    y_test  = sol_te.astype(np.float32)[:, :, :, None]

    print(
        f"\nDarcy .mat loaded  (r={r}, grid={h}x{h}):\n"
        f"  x_train={x_train.shape}  y_train={y_train.shape}\n"
        f"  x_test ={x_test.shape}   y_test ={y_test.shape}\n"
        f"  coeff range: [{coeff_tr.min():.3f}, {coeff_tr.max():.3f}]\n"
        f"  sol   range: [{sol_tr.min():.3f}, {sol_tr.max():.3f}]"
    )
    return x_train, y_train, x_test, y_test
