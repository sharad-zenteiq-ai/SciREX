"""
Standard data-driven loss functions for operator learning.
"""
import jax.numpy as jnp

def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error."""
    return jnp.mean((pred - target) ** 2)

def lp_loss(pred: jnp.ndarray, target: jnp.ndarray, p: int = 2) -> jnp.ndarray:
    """
    Relative Lp loss: ||pred - target||_p / ||target||_p
    Standard objective for operator learning.
    """
    # Flatten spatial and channel dimensions to (batch, -1)
    batch = pred.shape[0]
    diff = (pred - target).reshape(batch, -1)
    targ = target.reshape(batch, -1)
    
    # Compute norms per sample in batch
    diff_norm = jnp.linalg.norm(diff, ord=p, axis=1)
    targ_norm = jnp.linalg.norm(targ, ord=p, axis=1)
    
    # Return mean relative error over batch
    return jnp.mean(diff_norm / (targ_norm + 1e-8))
