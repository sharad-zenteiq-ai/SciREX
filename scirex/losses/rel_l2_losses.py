import jax.numpy as jnp
from scirex.losses.data_losses import lp_loss

def phys_rel_l2_loss(pred_encoded: jnp.ndarray, target_encoded: jnp.ndarray, normalizer) -> jnp.ndarray:
    """
    Computes the Relative L2 loss directly on the true physical domain signals.
    This is strictly equivalent to original FNO's `LpLoss(d=2)` behaviour where 
    it normalizes the loss over the un-normalized scale of the output targets!

    Args:
        pred_encoded: The model's raw forward-pass output (mean=0, std=1 if normalized)
        target_encoded: The ground truth output (mean=0, std=1 if normalized)
        normalizer: A `UnitGaussianNormalizer` instance that provides `.decode()`
    """
    # 1. Back-transform outputs to original unnormalized PDE scales
    pred_decoded = normalizer.decode(pred_encoded)
    target_decoded = normalizer.decode(target_encoded)

    # 2. Compute the lp_loss (Relative L2 norm) over the physical signals
    return lp_loss(pred_decoded, target_decoded, p=2)
