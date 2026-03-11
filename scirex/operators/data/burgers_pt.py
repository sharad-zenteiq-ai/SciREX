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
Loader for the neuraloperator 1-D Viscous Burgers .pt dataset.

Dataset layout (confirmed by torch + binary inspection)
-------------------------------------------------------
Each .pt file is a PyTorch checkpoint dict with three keys:

    'x'    : torch.Tensor  (N, nx)      float64
             Initial condition  u(x, t=0)

    'y'    : torch.Tensor  (N, T, nx)   float64   ← T-first axis order
             Full solution trajectory  u(x, t)  for T time steps

    'visc' : torch.Tensor  scalar       float64
             Kinematic viscosity  ν  (= 0.01 for these files)

File sizes for the 16-resolution files
    burgers_train_16.pt  →  N=1200,  nx=16,  T=17
    burgers_test_16.pt   →  N=1200,  nx=16,  T=17

IMPORTANT – axis order of y
    PyTorch stores y as (N, T, nx) i.e. time axis comes BEFORE spatial.
    The ZIP fallback reader matches this: y is reshaped as (N, T, nx).
    The final time step is therefore  y[:, -1, :]  (not y[:, :, -1]).

Operator learned by FNO2D
--------------------------
    u₀(x)  ──FNO2D──>  u(x, T_final)

Because the data is 1-D (one spatial axis), we add a trivial second
spatial dimension (ny=1) so the SciREX FNO2D module can process it
without any code changes:

    Input  shape: (N, nx, 1, 2)   channels = [u₀,  grid_x]
    Target shape: (N, nx, 1, 1)   u at the final time step  y[:, -1, :]

The grid coordinate is the only channel appended (no grid_y since ny=1).
This mirrors the convention in train_poisson2d_fno.py and darcy_pt.py.

PDE reference
-------------
    ∂u/∂t  +  u · ∂u/∂x  =  ν · ∂²u/∂x²
    x ∈ [0, 1]  (periodic),   t ∈ [0, T],   ν = 0.01
"""

from typing import Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal: raw .pt reader (no torch required at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _read_pt_without_torch(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Read a neuraloperator Burgers .pt file without importing PyTorch.

    The .pt format is a ZIP archive containing raw float64 blobs named
    ``data/0`` (x), ``data/1`` (y), ``data/2`` (visc).  The archive
    prefix (folder name inside the ZIP) is inferred automatically.

    Args:
        path : Path to the .pt file.

    Returns:
        x    : (N, nx)    float32  initial conditions
        y    : (N, nx, T) float32  full trajectories
        visc : float      kinematic viscosity
    """
    import zipfile

    with zipfile.ZipFile(path) as zf:
        # Infer archive prefix from first entry (e.g. 'burgers_16_train/')
        prefix = zf.namelist()[0].split("/")[0]

        with zf.open(f"{prefix}/data/0") as f:
            x_raw = f.read()
        with zf.open(f"{prefix}/data/1") as f:
            y_raw = f.read()
        with zf.open(f"{prefix}/data/2") as f:
            v_raw = f.read()

    # Blobs are stored as little-endian float64
    x_flat = np.frombuffer(x_raw, dtype=np.float64)
    y_flat = np.frombuffer(y_raw, dtype=np.float64)
    visc   = float(np.frombuffer(v_raw, dtype=np.float64)[0])

    # Infer N, nx, T from blob sizes
    # x: (N, nx)      y: (N, T, nx)  — time axis is FIRST (matches torch)
    # Verified: y_B[0, 0, :8] == x[0, :8]  (t=0 slice along axis 1)
    nx    = 16                                # spatial resolution from filename
    N     = len(x_flat) // nx
    T     = len(y_flat) // (N * nx)          # = 17

    x = x_flat.reshape(N, nx).astype(np.float32)
    y = y_flat.reshape(N, T, nx).astype(np.float32)   # (N, T, nx)

    return x, y, visc


def _read_pt_with_torch(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Read a neuraloperator Burgers .pt file using PyTorch (preferred).

    PyTorch returns y with shape (N, T, nx) — time axis is first.

    Args:
        path : Path to the .pt file.

    Returns:
        x    : (N, nx)    float32
        y    : (N, T, nx) float32   ← time-first axis order
        visc : float
    """
    import torch
    d = torch.load(path, map_location="cpu")
    x    = d["x"].numpy().astype(np.float32)
    y    = d["y"].numpy().astype(np.float32)   # (N, T, nx)
    visc = float(d["visc"].item())
    return x, y, visc


def _load_raw(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Try torch first; fall back to the ZIP reader if torch is absent."""
    try:
        return _read_pt_with_torch(path)
    except ImportError:
        return _read_pt_without_torch(path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_burgers_pt(
    train_path: str,
    test_path:  str,
    n_train:    int = 1000,
    n_test:     int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the neuraloperator 1-D Burgers .pt dataset and return arrays
    shaped for the SciREX FNO2D module.

    The loader adds a trivial second spatial dimension (ny = 1) so that
    the 1-D problem can pass through ``FNO2D`` unchanged.  A normalised
    spatial grid coordinate is appended as the second input channel,
    matching the 3-channel convention used in ``train_darcy_pt_fno2d.py``
    and ``train_poisson2d_fno.py``.

    Input channels:
        ch 0  –  initial condition  u₀(x)
        ch 1  –  normalised grid    x ∈ [0, 1]

    Target:
        ch 0  –  u(x, T_final)   (last time step of trajectory)

    Args:
        train_path : Path to ``burgers_train_16.pt``.
        test_path  : Path to ``burgers_test_16.pt``.
        n_train    : Samples to use from train file  (≤ file size).
        n_test     : Samples to use from test  file  (≤ file size).

    Returns:
        x_train : (n_train, nx, 1, 2)  float32
        y_train : (n_train, nx, 1, 1)  float32
        x_test  : (n_test,  nx, 1, 2)  float32
        y_test  : (n_test,  nx, 1, 1)  float32
    """
    # ── Load raw tensors ──────────────────────────────────────────────────
    x_tr_raw, y_tr_raw, visc_tr = _load_raw(train_path)
    x_te_raw, y_te_raw, visc_te = _load_raw(test_path)

    # Clip to requested subset
    x_tr_raw = x_tr_raw[:n_train]          # (n_train, nx)
    y_tr_raw = y_tr_raw[:n_train]          # (n_train, T, nx)
    x_te_raw = x_te_raw[:n_test]           # (n_test,  nx)
    y_te_raw = y_te_raw[:n_test]           # (n_test,  T, nx)

    nx = x_tr_raw.shape[1]                 # 16

    # ── Build normalised grid coordinate ─────────────────────────────────
    grid = np.linspace(0.0, 1.0, nx, dtype=np.float32)   # (nx,)

    def _to_fno2d(x_1d: np.ndarray, y_full: np.ndarray):
        """
        Convert raw (N, nx) IC and (N, T, nx) trajectory to FNO2D format.

        y_full has shape (N, T, nx) — time axis is dim-1 (torch convention).
        Final time step is therefore  y_full[:, -1, :]  (shape: N, nx).

        Returns:
            x_out : (N, nx, 1, 2)   [u₀, grid_x]
            y_out : (N, nx, 1, 1)   u at final time step
        """
        N = x_1d.shape[0]

        # IC channel: (N, nx) -> (N, nx, 1, 1)
        u0 = x_1d[:, :, np.newaxis, np.newaxis]

        # Grid channel: (nx,) -> (N, nx, 1, 1)
        gx = np.tile(grid[np.newaxis, :, np.newaxis, np.newaxis], (N, 1, 1, 1))

        # Input: concatenate along channel axis → (N, nx, 1, 2)
        x_out = np.concatenate([u0, gx], axis=-1)

        # Target: final time step — y is (N, T, nx) so last step is axis-1 index -1
        y_final = y_full[:, -1, :]                          # (N, nx)
        y_out   = y_final[:, :, np.newaxis, np.newaxis]     # (N, nx, 1, 1)

        return x_out.astype(np.float32), y_out.astype(np.float32)

    x_train, y_train = _to_fno2d(x_tr_raw, y_tr_raw)
    x_test,  y_test  = _to_fno2d(x_te_raw, y_te_raw)

    print(
        f"Burgers dataset loaded  (ν = {visc_tr:.4f}):\n"
        f"  x_train={x_train.shape}   y_train={y_train.shape}\n"
        f"  x_test ={x_test.shape}    y_test ={y_test.shape}\n"
        f"  u₀ range : [{x_tr_raw.min():.4f}, {x_tr_raw.max():.4f}]\n"
        f"  uT range : [{y_train[..., 0].min():.4f}, {y_train[..., 0].max():.4f}]"
    )
    return x_train, y_train, x_test, y_test
