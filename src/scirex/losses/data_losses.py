"""
Standard data-driven loss functions for operator learning.
"""
import jax.numpy as jnp

def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error."""
    return jnp.mean((pred - target) ** 2)

def lp_loss(pred: jnp.ndarray, target: jnp.ndarray, p: int = 2) -> jnp.ndarray:
    """Relative Lp loss (common in FNO literature)."""
    # Not implemented yet, place-holder for future physics/data losses
    return jnp.mean(jnp.abs(pred - target) ** p)
