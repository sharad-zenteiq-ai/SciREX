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
Poisson 3D dataset generator (periodic domain) using FFT-based Poisson solver.

Generates batches of RHS f(x,y,z) (smooth random low-frequency fields) and
computes the corresponding solution u(x,y,z) of Laplace(u) = f with periodic BC
by inverting the Laplacian in Fourier space.

Follows the same technique as scirex/data/datasets/poisson.py but extended to 3D.
"""
from typing import Iterator, Tuple
import numpy as np


def solve_poisson_periodic_batch_3d(f_batch: np.ndarray) -> np.ndarray:
    """
    Solve Poisson for a batch of RHS f on a 3D periodic domain.

    f_batch: (batch, nx, ny, nz) or (batch, nx, ny, nz, 1)
    returns: u_batch same shape as f_batch (with channel dim)
    """
    f = f_batch
    if f.ndim == 5 and f.shape[-1] == 1:
        f = f[..., 0]
    batch, nx, ny, nz = f.shape
    u = np.zeros_like(f, dtype=np.float32)

    # Precompute wavenumbers
    kx = np.fft.fftfreq(nx, d=1.0 / nx) * 2.0 * np.pi  # shape (nx,)
    ky = np.fft.fftfreq(ny, d=1.0 / ny) * 2.0 * np.pi  # shape (ny,)
    kz = np.fft.fftfreq(nz, d=1.0 / nz) * 2.0 * np.pi  # shape (nz,)
    
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kx3d ** 2 + ky3d ** 2 + kz3d ** 2
    
    # Avoid divide-by-zero at zero frequency
    k2[0, 0, 0] = 1.0

    for i in range(batch):
        F_hat = np.fft.fftn(f[i])
        U_hat = -F_hat / k2
        U_hat[0, 0, 0] = 0.0  # set mean to zero
        ui = np.fft.ifftn(U_hat).real
        u[i] = ui.astype(np.float32)
        
    # Add channel dim if it was present or requested
    return u[..., np.newaxis]


def random_poisson_3d_batch(
    batch_size: int, 
    nx: int, 
    ny: int, 
    nz: int, 
    channels: int = 1, 
    rng_seed: int = 0, 
    max_modes: int = 3,
    include_mesh: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single batch of (f, u) pairs in 3D.

    f is generated as a sum of a few low-frequency sinusoids with random
    amplitudes/phases. u is computed via FFT Poisson solve.

    Args:
        batch_size: Number of samples in batch.
        nx, ny, nz: Grid resolution.
        channels: Number of solution channels (default 1).
        rng_seed: Random seed for reproducibility.
        max_modes: Maximum number of frequency modes to sum.
        include_mesh: If True, returns f concatenated with (x,y,z) coordinates.

    Returns:
      f_batch: (batch, nx, ny, nz, 1 or 4) float32
      u_batch: (batch, nx, ny, nz, 1) float32
    """
    rng = np.random.default_rng(rng_seed)
    xs = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    ys = np.linspace(0, 2 * np.pi, ny, endpoint=False)
    zs = np.linspace(0, 2 * np.pi, nz, endpoint=False)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # For include_mesh, we'll need coordinates in [0, 1]
    xs_norm = np.linspace(0, 1, nx)
    ys_norm = np.linspace(0, 1, ny)
    zs_norm = np.linspace(0, 1, nz)
    X_norm, Y_norm, Z_norm = np.meshgrid(xs_norm, ys_norm, zs_norm, indexing="ij")

    f_batch_pure = np.zeros((batch_size, nx, ny, nz, 1), dtype=np.float32)
    
    for b in range(batch_size):
        field = np.zeros((nx, ny, nz), dtype=np.float32)
        nmodes = rng.integers(1, max_modes + 1)
        for _ in range(nmodes):
            ax = rng.integers(1, max(2, nx // 4))
            ay = rng.integers(1, max(2, ny // 4))
            az = rng.integers(1, max(2, nz // 4))
            amp = float(rng.normal(0, 1.0))
            phase = rng.uniform(0, 2 * np.pi)
            field += amp * np.sin(ax * X + ay * Y + az * Z + phase)
            
        std = np.std(field)
        if std > 0:
            field = field / std * 1.0
        f_batch_pure[b, ..., 0] = field

    u_batch = solve_poisson_periodic_batch_3d(f_batch_pure) * 1000
    
    if include_mesh:
        # Create coordinates batch: (batch, nx, ny, nz, 3)
        X_c = np.tile(X_norm[np.newaxis, ..., np.newaxis], (batch_size, 1, 1, 1, 1))
        Y_c = np.tile(Y_norm[np.newaxis, ..., np.newaxis], (batch_size, 1, 1, 1, 1))
        Z_c = np.tile(Z_norm[np.newaxis, ..., np.newaxis], (batch_size, 1, 1, 1, 1))
        
        # Concatenate f with x, y, z
        f_batch = np.concatenate([f_batch_pure, X_c, Y_c, Z_c], axis=-1)
    else:
        f_batch = f_batch_pure

    return f_batch.astype(np.float32), u_batch.astype(np.float32)


def generator(
    num_batches: int,
    batch_size: int,
    nx: int,
    ny: int,
    nz: int,
    channels: int = 1,
    rng_seed: int = 0,
    include_mesh: bool = False
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields num_batches batches of (f, u) pairs.
    """
    for i in range(num_batches):
        f, u = random_poisson_3d_batch(
            batch_size, nx, ny, nz, channels, rng_seed=rng_seed + i, include_mesh=include_mesh
        )
        yield f, u
