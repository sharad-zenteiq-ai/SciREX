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
Poisson 2D dataset generator (periodic domain) using FFT-based Poisson solver.

Generates batches of RHS f(x,y) (smooth random low-frequency fields) and
computes the corresponding solution u(x,y) of Laplace(u) = f with periodic BC
by inverting the Laplacian in Fourier space.

Notes:
- Domain is periodic on [0,1)x[0,1).
- The k=0 Fourier mode (mean) is set to zero to ensure solvability.
- Returns numpy arrays (float32); convert to jnp when feeding the model.
"""
from typing import Iterator, Tuple
import numpy as np


def solve_poisson_periodic_batch(f_batch: np.ndarray) -> np.ndarray:
    """
    Solve Poisson for a batch of RHS f on periodic domain.

    f_batch: (batch, nx, ny) or (batch, nx, ny, 1)
    returns: u_batch same shape as f_batch (without channel dim if input lacked it)
    """
    f = f_batch
    if f.ndim == 4 and f.shape[-1] == 1:
        f = f[..., 0]
    batch, nx, ny = f.shape
    u = np.zeros_like(f, dtype=np.float32)

    # Precompute wavenumbers
    kx = np.fft.fftfreq(nx, d=1.0 / nx) * 2.0 * np.pi  # shape (nx,)
    ky = np.fft.fftfreq(ny, d=1.0 / ny) * 2.0 * np.pi  # shape (ny,)
    kx2d, ky2d = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx2d ** 2 + ky2d ** 2
    # Avoid divide-by-zero at zero frequency
    k2[0, 0] = 1.0

    for i in range(batch):
        F_hat = np.fft.fft2(f[i])
        U_hat = -F_hat / k2
        U_hat[0, 0] = 0.0  # set mean to zero (or any constant)
        ui = np.fft.ifft2(U_hat).real
        u[i] = ui.astype(np.float32)
    # Add channel dim
    return u[..., np.newaxis]


def random_poisson_batch(
    batch_size: int, nx: int, ny: int, channels: int = 1, rng_seed: int = 0, max_modes: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single batch of (f, u) pairs.

    f is generated as a sum of a few low-frequency sinusoids with random
    amplitudes/phases to produce smooth RHS fields. u is computed via FFT Poisson solve.

    Returns:
      f_batch: (batch, nx, ny, channels) float32
      u_batch: (batch, nx, ny, channels) float32
    """
    rng = np.random.default_rng(rng_seed)
    xs = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    ys = np.linspace(0, 2 * np.pi, ny, endpoint=False)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    f_batch = np.zeros((batch_size, nx, ny, channels), dtype=np.float32)
    
    # Precompute wavenumbers for GRF
    # Usually: alpha=2.0, tau=3.0 for 2D FNO Poisson
    k_max = nx // 2
    kx = np.fft.fftfreq(nx, d=1.0) * nx
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    k_sq = Kx**2 + Ky**2
    
    alpha = 2.0
    tau = 3.0
    # Inverse square root of eigenvalues of covariance
    inv_eigen = 1.0 / ((k_sq + tau**2) ** alpha)
    inv_eigen[0, 0] = 0.0 # Zero mean
    
    for b in range(batch_size):
        # Sample random noise in Fourier space
        noise = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
        F_hat = noise * inv_eigen * nx * ny
        field = np.fft.ifft2(F_hat).real
        
        # normalize
        std = np.std(field)
        if std > 0:
            field = field / std * 1.0
        f_batch[b, :, :, 0] = field
    
    # Create normalized coordinates: (nx, ny, 2)
    xs_norm = np.linspace(0, 1, nx)
    ys_norm = np.linspace(0, 1, ny)
    X_norm, Y_norm = np.meshgrid(xs_norm, ys_norm, indexing="ij")
    
    # Broadcast to (batch, nx, ny, 2)
    grid_x = np.tile(X_norm[None, ..., None], (batch_size, 1, 1, 1))
    grid_y = np.tile(Y_norm[None, ..., None], (batch_size, 1, 1, 1))
    
    # Concatenate f with x, y: final shape (batch, nx, ny, 3)
    f_batch = np.concatenate([f_batch, grid_x, grid_y], axis=-1)
    
    # Target solution u 
    u_batch = solve_poisson_periodic_batch(f_batch[..., :1])
        
    return np.asarray(f_batch).astype(np.float32), np.asarray(u_batch).astype(np.float32)


def generator(
    num_batches: int,
    batch_size: int,
    nx: int,
    ny: int,
    channels: int = 1, # Base channels (will be appended with coordinates)
    rng_seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields num_batches batches of (f, u) pairs.
    """
    for i in range(num_batches):
        f, u = random_poisson_batch(batch_size, nx, ny, channels, rng_seed=rng_seed + i)
        yield f, u
