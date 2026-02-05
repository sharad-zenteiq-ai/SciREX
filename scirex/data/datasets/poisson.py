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
    for b in range(batch_size):
        field = np.zeros((nx, ny), dtype=np.float32)
        # sum of a few low-frequency sine/cosine modes
        nmodes = rng.integers(1, max_modes + 1)
        for _ in range(nmodes):
            ax = rng.integers(1, max(2, nx // 4))
            ay = rng.integers(1, max(2, ny // 4))
            amp = float(rng.normal(0, 1.0))
            phase = rng.uniform(0, 2 * np.pi)
            field += amp * np.sin(ax * X + ay * Y + phase)
        # normalize
        std = np.std(field)
        if std > 0:
            field = field / std * 1.0
        f_batch[b, :, :, 0] = field

    u_batch = solve_poisson_periodic_batch(f_batch) * 1000
    return f_batch.astype(np.float32), u_batch.astype(np.float32)


def generator(
    num_batches: int,
    batch_size: int,
    nx: int,
    ny: int,
    channels: int = 1,
    rng_seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields num_batches batches of (f, u) pairs.
    """
    for i in range(num_batches):
        f, u = random_poisson_batch(batch_size, nx, ny, channels, rng_seed=rng_seed + i)
        yield f, u
