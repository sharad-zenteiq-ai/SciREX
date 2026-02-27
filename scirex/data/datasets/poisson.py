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
Poisson dataset generators (periodic domain) using FFT-based Poisson solver.
Supports 2D and 3D domains.
"""
from typing import Iterator, Tuple
import numpy as np


# ────────────────────────────────────────────────────────────────────
# 2D POISSON
# ────────────────────────────────────────────────────────────────────

def solve_poisson_periodic_batch_2d(f_batch: np.ndarray) -> np.ndarray:
    """
    Computes the solution 'u' to the 2D Poisson equation -∇²u = f on a periodic domain.
    
    The solver utilizes the Spectral Method by performing a Fast Fourier Transform (FFT) 
    on the source term, applying the inverse Laplacian in the frequency domain, 
    and returning the result to the spatial domain via an Inverse FFT.

    Args:
        f_batch (np.ndarray): Source term(s) of shape (batch, nx, ny) or (batch, nx, ny, 1).
        
    Returns:
        np.ndarray: Solution field 'u' of shape (batch, nx, ny, 1).
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
        U_hat[0, 0] = 0.0  # set mean to zero
        ui = np.fft.ifft2(U_hat).real
        u[i] = ui.astype(np.float32)
    # Add channel dim
    return u[..., np.newaxis]


def random_poisson_2d_batch(
    batch_size: int, nx: int, ny: int, channels: int = 1, rng_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes a batch of random 2D Poisson samples using Gaussian Random Fields (GRF).
    
    The source term 'f' is generated as a GRF with a specific spectral decay, 
    ensuring spatial correlations. The corresponding solution 'u' is then 
    exactly computed via the spectral Poisson solver.
    """
    rng = np.random.default_rng(rng_seed)
    
    f_batch_pure = np.zeros((batch_size, nx, ny, 1), dtype=np.float32)
    
    # Precompute wavenumbers for GRF
    kx = np.fft.fftfreq(nx, d=1.0) * nx
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    Kx, Ky = np.meshgrid(kx, ky, indexing="ij")
    k_sq = Kx**2 + Ky**2
    
    alpha = 2.0
    tau = 3.0
    inv_eigen = 1.0 / ((k_sq + tau**2) ** alpha)
    inv_eigen[0, 0] = 0.0 # Zero mean
    
    for b in range(batch_size):
        noise = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))
        F_hat = noise * inv_eigen * nx * ny
        field = np.fft.ifft2(F_hat).real
        
        std = np.std(field)
        if std > 0:
            field = field / std * 1.0
        f_batch_pure[b, :, :, 0] = field
    
    # Create normalized coordinates
    xs_norm = np.linspace(0, 1, nx)
    ys_norm = np.linspace(0, 1, ny)
    X_norm, Y_norm = np.meshgrid(xs_norm, ys_norm, indexing="ij")
    
    # Broadcast to coordinates
    grid_x = np.tile(X_norm[None, ..., None], (batch_size, 1, 1, 1))
    grid_y = np.tile(Y_norm[None, ..., None], (batch_size, 1, 1, 1))
    
    # Concatenate f with x, y
    f_batch = np.concatenate([f_batch_pure, grid_x, grid_y], axis=-1)
    
    # Target solution u 
    u_batch = solve_poisson_periodic_batch_2d(f_batch_pure)
        
    return f_batch.astype(np.float32), u_batch.astype(np.float32)


def generator_2d(
    num_batches: int,
    batch_size: int,
    nx: int,
    ny: int,
    channels: int = 1,
    rng_seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields num_batches batches of 2D (f, u) pairs.
    """
    for i in range(num_batches):
        f, u = random_poisson_2d_batch(batch_size, nx, ny, channels, rng_seed=rng_seed + i)
        yield f, u


# ────────────────────────────────────────────────────────────────────
# 3D POISSON
# ────────────────────────────────────────────────────────────────────

def solve_poisson_periodic_batch_3d(f_batch: np.ndarray) -> np.ndarray:
    """
    Computes the solution 'u' to the 3D Poisson equation -∇²u = f on a periodic domain.
    
    Implementation mirrors the spectral 2D solver, extended to volumetric data.
    """
    f = f_batch
    if f.ndim == 5 and f.shape[-1] == 1:
        f = f[..., 0]
    batch, nx, ny, nz = f.shape
    u = np.zeros_like(f, dtype=np.float32)

    # Precompute wavenumbers
    kx = np.fft.fftfreq(nx, d=1.0 / nx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=1.0 / ny) * 2.0 * np.pi
    kz = np.fft.fftfreq(nz, d=1.0 / nz) * 2.0 * np.pi
    
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kx3d ** 2 + ky3d ** 2 + kz3d ** 2
    
    # Avoid divide-by-zero at zero frequency
    k2[0, 0, 0] = 1.0

    for i in range(batch):
        F_hat = np.fft.fftn(f[i])
        U_hat = -F_hat / k2
        U_hat[0, 0, 0] = 0.0
        ui = np.fft.ifftn(U_hat).real
        u[i] = ui.astype(np.float32)
        
    return u[..., np.newaxis]


def random_poisson_3d_batch(
    batch_size: int, 
    nx: int, 
    ny: int, 
    nz: int, 
    channels: int = 1, 
    rng_seed: int = 0,
    include_mesh: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single batch of (f, u) pairs in 3D.
    """
    rng = np.random.default_rng(rng_seed)
    
    f_batch_pure = np.zeros((batch_size, nx, ny, nz, 1), dtype=np.float32)
    
    # Precompute wavenumbers for GRF
    kx = np.fft.fftfreq(nx, d=1.0) * nx
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    kz = np.fft.fftfreq(nz, d=1.0) * nz
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing="ij")
    k_sq = Kx**2 + Ky**2 + Kz**2
    
    alpha = 2.0
    tau = 3.0
    inv_eigen = 1.0 / ((k_sq + tau**2) ** alpha)
    inv_eigen[0, 0, 0] = 0.0 # Zero mean
    
    for b in range(batch_size):
        noise = rng.normal(size=(nx, ny, nz)) + 1j * rng.normal(size=(nx, ny, nz))
        F_hat = noise * inv_eigen * nx * ny * nz
        field = np.fft.ifftn(F_hat).real
        
        std = np.std(field)
        if std > 0:
            field = field / std * 1.0
        f_batch_pure[b, ..., 0] = field

    u_batch = solve_poisson_periodic_batch_3d(f_batch_pure)
    
    if include_mesh:
        xs_norm = np.linspace(0, 1, nx)
        ys_norm = np.linspace(0, 1, ny)
        zs_norm = np.linspace(0, 1, nz)
        X_norm, Y_norm, Z_norm = np.meshgrid(xs_norm, ys_norm, zs_norm, indexing="ij")
        
        X_c = np.tile(X_norm[np.newaxis, ..., np.newaxis], (batch_size, 1, 1, 1, 1))
        Y_c = np.tile(Y_norm[np.newaxis, ..., np.newaxis], (batch_size, 1, 1, 1, 1))
        Z_c = np.tile(Z_norm[np.newaxis, ..., np.newaxis], (batch_size, 1, 1, 1, 1))
        
        f_batch = np.concatenate([f_batch_pure, X_c, Y_c, Z_c], axis=-1)
    else:
        f_batch = f_batch_pure

    return f_batch.astype(np.float32), u_batch.astype(np.float32)


def generator_3d(
    num_batches: int,
    batch_size: int,
    nx: int,
    ny: int,
    nz: int,
    channels: int = 1,
    rng_seed: int = 0,
    include_mesh: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields num_batches batches of 3D (f, u) pairs.
    """
    for i in range(num_batches):
        f, u = random_poisson_3d_batch(
            batch_size, nx, ny, nz, channels, rng_seed=rng_seed + i, include_mesh=include_mesh
        )
        yield f, u

# Aliases for backward compatibility
solve_poisson_periodic_batch = solve_poisson_periodic_batch_2d
random_poisson_batch = random_poisson_2d_batch
generator = generator_2d
