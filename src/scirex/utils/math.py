"""
Shared mathematical utilities for FNO operations.
"""
from typing import Tuple
import jax.numpy as jnp

def rfft2(x: jnp.ndarray) -> jnp.ndarray:
    """2D real-to-complex FFT along axes (1,2): (batch, nx, ny, ch) -> (..., ny//2+1, ch) complex"""
    return jnp.fft.rfft2(x, axes=(1, 2))

def irfft2(X: jnp.ndarray, s: Tuple[int, int]) -> jnp.ndarray:
    """Inverse 2D real FFT with explicit output spatial shape s=(nx, ny)."""
    return jnp.fft.irfft2(X, s=s, axes=(1, 2))
