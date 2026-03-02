import jax
import jax.numpy as jnp
import pytest
from scirex.operators.utils.math import rfft2, irfft2

def test_rfft2_irfft2_consistency():
    # Create dummy data (batch, nx, ny, channels)
    key = jax.random.PRNGKey(0)
    nx, ny = 16, 16
    x = jax.random.normal(key, (2, nx, ny, 3))
    
    # Forward FFT
    X = rfft2(x)
    
    # Expected shape: (batch, nx, ny//2 + 1, channels)
    assert X.shape == (2, 16, 9, 3)
    
    # Inverse FFT
    x_rec = irfft2(X, s=(nx, ny))
    
    # Check shape and values
    assert x_rec.shape == (2, 16, 16, 3)
    assert jnp.allclose(x, x_rec, atol=1e-5)

def test_rfft2_shapes():
    x = jnp.ones((1, 32, 32, 1))
    X = rfft2(x)
    assert X.shape == (1, 32, 17, 1)
    
    x_rec = irfft2(X, s=(32, 32))
    assert x_rec.shape == (1, 32, 32, 1)
