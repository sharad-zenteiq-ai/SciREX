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
2D Darcy flow dataset generator.

Solves: -div(a(x,y) grad u(x,y)) = f(x,y) with Dirichlet BCs (u=0 on boundary).

This module provides two modes for generating permeability fields:
1. 'binary' (FNO-style): Thresholded GRF with values {a_low, a_high} (default: {4, 12})
   - Matches the original FNO paper (Li et al., 2020)
   - GRF covariance: C = (-Δ + τ²I)^(-α) with α=2, τ=3 (corresponds to Matérn-like)
   
2. 'continuous': Log-normal permeability field a = exp(GRF)
   - Smoother, more physical in some applications

Input: permeability field a(x,y)
Output: pressure field u(x,y)
"""
import numpy as np
from typing import Iterator, Tuple, Literal

# =============================================================================
# Gaussian Random Field Generator (FNO-style)
# =============================================================================

def generate_grf_2d_fno(
    nx: int, 
    ny: int, 
    alpha: float = 2.0, 
    tau: float = 3.0, 
    rng=None
) -> np.ndarray:
    """
    Generate a 2D Gaussian Random Field with covariance operator:
        C = (-Δ + τ²I)^(-α)
    
    This matches the FNO paper's data generation for Darcy flow.
    In Fourier space: S(k) = (|k|² + τ²)^(-α)
    
    Args:
        nx, ny: Grid resolution
        alpha: Smoothness parameter (higher = smoother). FNO uses α=2.
        tau: Inverse length scale. FNO uses τ=3.
        rng: NumPy random generator
        
    Returns:
        field: (nx, ny) array, normalized to zero mean and unit variance
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Wavenumbers (scaled for unit domain [0,1]²)
    # k = 2π * n where n is the frequency index
    kx = np.fft.fftfreq(nx) * nx * 2 * np.pi
    ky = np.fft.fftfreq(ny) * ny * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    # Power spectrum: S(k) = (|k|² + τ²)^(-α)
    # This corresponds to covariance operator C = (-Δ + τ²I)^(-α)
    spectrum = (K2 + tau**2)**(-alpha)
    spectrum[0, 0] = 0  # Remove DC component (set mean to zero)
    
    # Generate complex white noise and scale by sqrt(spectrum)
    noise_real = rng.standard_normal((nx, ny))
    noise_imag = rng.standard_normal((nx, ny))
    noise = noise_real + 1j * noise_imag
    
    field_hat = noise * np.sqrt(spectrum)
    
    # Transform to physical space
    field = np.fft.ifft2(field_hat).real
    
    # Normalize to zero mean, unit variance
    field = (field - np.mean(field)) / (np.std(field) + 1e-8)
    
    return field.astype(np.float32)


def threshold_permeability(
    grf: np.ndarray,
    threshold: float = 0.0,
    a_low: float = 4.0,
    a_high: float = 12.0
) -> np.ndarray:
    """
    Convert a GRF to binary permeability field by thresholding.
    
    This is the approach used in the FNO paper:
        a(x) = a_low  if GRF(x) < threshold
        a(x) = a_high if GRF(x) >= threshold
    
    Args:
        grf: Gaussian random field (typically normalized)
        threshold: Threshold value (0 gives ~50/50 split for normalized GRF)
        a_low: Permeability value below threshold (FNO uses 4 or 3)
        a_high: Permeability value above threshold (FNO uses 12)
        
    Returns:
        Binary permeability field with values {a_low, a_high}
    """
    return np.where(grf < threshold, a_low, a_high).astype(np.float32)


# =============================================================================
# Darcy PDE Solver
# =============================================================================

def solve_darcy_2d(
    a: np.ndarray, 
    f: np.ndarray, 
    max_iter: int = 10000, 
    tol: float = 1e-8
) -> np.ndarray:
    """
    Solve -div(a grad u) = f with Dirichlet boundary conditions (u=0).
    
    Uses Conjugate Gradient method for the symmetric positive definite system.
    
    Args:
        a: (nx, ny) permeability field (must be positive)
        f: (nx, ny) source term
        max_iter: Maximum CG iterations
        tol: Convergence tolerance for residual norm
        
    Returns:
        u: (nx, ny) solution field
    """
    nx, ny = a.shape
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    u = np.zeros_like(f, dtype=np.float64)  # Use float64 for solver stability
    a = a.astype(np.float64)
    f = f.astype(np.float64)
    
    def apply_A(v):
        """Apply the discrete operator -div(a grad v) with zero BCs."""
        Av = np.zeros_like(v)
        
        # Harmonic mean of permeability at cell faces (more stable for high contrast)
        # a_{i+1/2} = 2 * a_i * a_{i+1} / (a_i + a_{i+1})
        eps = 1e-10
        ax = 2.0 * a[1:, :] * a[:-1, :] / (a[1:, :] + a[:-1, :] + eps)
        ay = 2.0 * a[:, 1:] * a[:, :-1] / (a[:, 1:] + a[:, :-1] + eps)
        
        # Fluxes: a * (u_{i+1} - u_i) / dx
        flux_x = ax * (v[1:, :] - v[:-1, :]) / dx
        flux_y = ay * (v[:, 1:] - v[:, :-1]) / dy
        
        # Divergence of flux (negative because we have -div(a grad u))
        # At interior points only
        Av[1:-1, 1:-1] = -(
            (flux_x[1:, 1:-1] - flux_x[:-1, 1:-1]) / dx +
            (flux_y[1:-1, 1:] - flux_y[1:-1, :-1]) / dy
        )
        
        return Av
    
    # CG iteration
    r = f.copy()
    r[0, :] = 0; r[-1, :] = 0; r[:, 0] = 0; r[:, -1] = 0  # Zero BC in RHS
    r = r - apply_A(u)
    
    p = r.copy()
    rsold = np.sum(r**2)
    
    if rsold < tol**2:
        return u.astype(np.float32)
    
    for iteration in range(max_iter):
        Ap = apply_A(p)
        pAp = np.sum(p * Ap)
        
        if abs(pAp) < 1e-14:
            break
            
        alpha = rsold / pAp
        u = u + alpha * p
        r = r - alpha * Ap
        
        rsnew = np.sum(r**2)
        
        if np.sqrt(rsnew) < tol:
            break
            
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return u.astype(np.float32)


# =============================================================================
# Batch Data Generation
# =============================================================================

def random_darcy_batch(
    batch_size: int, 
    nx: int, 
    ny: int, 
    rng_seed: int = 0,
    mode: Literal["binary", "continuous"] = "binary",
    alpha: float = 2.0,
    tau: float = 3.0,
    a_low: float = 4.0,
    a_high: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of (permeability, pressure) pairs for Darcy flow.
    
    Args:
        batch_size: Number of samples
        nx, ny: Grid resolution
        rng_seed: Random seed for reproducibility
        mode: 'binary' for FNO-style thresholded GRF, 'continuous' for log-normal
        alpha, tau: GRF parameters (FNO uses α=2, τ=3)
        a_low, a_high: Binary mode permeability values (FNO uses 4, 12 or 3, 12)
        
    Returns:
        a_batch: (batch, nx, ny, 1) permeability fields
        u_batch: (batch, nx, ny, 1) pressure solutions
    """
    rng = np.random.default_rng(rng_seed)
    
    a_batch = np.zeros((batch_size, nx, ny, 1), dtype=np.float32)
    u_batch = np.zeros((batch_size, nx, ny, 1), dtype=np.float32)
    
    # Constant source term f = 1 (standard in FNO benchmarks)
    f = np.ones((nx, ny), dtype=np.float32)
    
    for b in range(batch_size):
        # Generate GRF
        grf = generate_grf_2d_fno(nx, ny, alpha=alpha, tau=tau, rng=rng)
        
        # Convert to permeability based on mode
        if mode == "binary":
            # FNO-style: threshold to get binary {a_low, a_high}
            a = threshold_permeability(grf, threshold=0.0, a_low=a_low, a_high=a_high)
        else:
            # Continuous: log-normal permeability
            a = np.exp(grf)
        
        # Solve PDE
        u = solve_darcy_2d(a, f)
        
        a_batch[b, ..., 0] = a
        u_batch[b, ..., 0] = u * 100.0  # Scale for training stability
        
    return a_batch, u_batch


def generator(
    num_batches: int,
    batch_size: int,
    nx: int,
    ny: int,
    rng_seed: int = 0,
    mode: Literal["binary", "continuous"] = "binary",
    alpha: float = 2.0,
    tau: float = 3.0,
    a_low: float = 4.0,
    a_high: float = 12.0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield batches of (permeability, pressure) pairs.
    
    Args:
        num_batches: Number of batches to generate
        batch_size: Samples per batch
        nx, ny: Grid resolution
        rng_seed: Base random seed (incremented per batch)
        mode: 'binary' (FNO-style) or 'continuous' (log-normal)
        alpha, tau: GRF parameters
        a_low, a_high: Binary permeability values
        
    Yields:
        (a_batch, u_batch) tuples
    """
    for i in range(num_batches):
        a, u = random_darcy_batch(
            batch_size, nx, ny, 
            rng_seed=rng_seed + i,
            mode=mode,
            alpha=alpha,
            tau=tau,
            a_low=a_low,
            a_high=a_high
        )
        yield a, u
