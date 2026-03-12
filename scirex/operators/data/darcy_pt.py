# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

"""
Loader for the original neuraloperator Darcy Flow .pt dataset.

The .pt files (e.g. darcy_train_16.pt / darcy_test_16.pt) are PyTorch
checkpoint dictionaries with keys:
    'x'  : torch.Tensor  (N, H, W)  — permeability field  a(x,y)
    'y'  : torch.Tensor  (N, H, W)  — pressure solution   u(x,y)

This module converts them to JAX-compatible numpy arrays and appends
normalised (x, y) grid coordinates so the FNO2D input is
    shape  (N, H, W, 3)  =  [a,  grid_x,  grid_y]
which is identical to the convention used throughout SciREX.

Reference dataset:
    Li et al. (2020) "Fourier Neural Operator for Parametric PDEs"
    https://github.com/neuraloperator/neuraloperator
"""

from typing import Tuple
import numpy as np


def load_darcy_pt(
    train_path: str,
    test_path: str,
    n_train: int = 1000,
    n_test: int = 200,
    resolution: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the neuraloperator Darcy .pt files and return numpy arrays.

    The loader appends normalised (x, y) grid coordinates to the
    permeability channel, producing 3-channel inputs that match the
    SciREX FNO2D convention used in ``train_poisson2d_fno.py``.

    Args:
        train_path  : Absolute or relative path to ``darcy_train_<res>.pt``.
        test_path   : Absolute or relative path to ``darcy_test_<res>.pt``.
        n_train     : Number of training samples to load  (≤ dataset size).
        n_test      : Number of test samples to load      (≤ dataset size).
        resolution  : Spatial resolution (used for the grid; inferred from
                      data if the loaded tensor size differs).

    Returns:
        x_train : (n_train, H, W, 3)  float32  [a, grid_x, grid_y]
        y_train : (n_train, H, W, 1)  float32  pressure u
        x_test  : (n_test,  H, W, 3)  float32
        y_test  : (n_test,  H, W, 1)  float32
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load .pt dataset files.\n"
            "Install it with:  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    # ── Load raw tensors ──────────────────────────────────────────────────
    train_dict = torch.load(train_path, map_location="cpu")
    test_dict  = torch.load(test_path,  map_location="cpu")

    # Keys 'x' (permeability) and 'y' (pressure), shape (N, H, W)
    a_train = train_dict["x"][:n_train].numpy().astype(np.float32)   # (N, H, W)
    u_train = train_dict["y"][:n_train].numpy().astype(np.float32)
    a_test  = test_dict["x"][:n_test].numpy().astype(np.float32)
    u_test  = test_dict["y"][:n_test].numpy().astype(np.float32)

    # Infer actual grid size from loaded data
    nx, ny = a_train.shape[1], a_train.shape[2]

    # ── Build normalised grid coordinates ─────────────────────────────────
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")          # (H, W)

    def _add_grid_and_channel(a: np.ndarray) -> np.ndarray:
        """(N, H, W) → (N, H, W, 3)  by appending grid channels."""
        N = a.shape[0]
        a_ch    = a[:, :, :, np.newaxis]                                    # (N,H,W,1)
        grid_x  = np.tile(GX[np.newaxis, :, :, np.newaxis], (N, 1, 1, 1))  # (N,H,W,1)
        grid_y  = np.tile(GY[np.newaxis, :, :, np.newaxis], (N, 1, 1, 1))  # (N,H,W,1)
        return np.concatenate([a_ch, grid_x, grid_y], axis=-1)              # (N,H,W,3)

    x_train = _add_grid_and_channel(a_train)
    x_test  = _add_grid_and_channel(a_test)

    y_train = u_train[:, :, :, np.newaxis]   # (N, H, W, 1)
    y_test  = u_test[:, :, :, np.newaxis]

    print(
        f"Darcy dataset loaded:\n"
        f"  x_train={x_train.shape}  y_train={y_train.shape}\n"
        f"  x_test ={x_test.shape}   y_test ={y_test.shape}\n"
        f"  a range: [{a_train.min():.3f}, {a_train.max():.3f}]   "
        f"  u range: [{u_train.min():.4f}, {u_train.max():.4f}]"
    )
    return x_train, y_train, x_test, y_test
