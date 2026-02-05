from flax import linen as nn
import jax.numpy as jnp

class Lifting(nn.Module):
    """
    Lifting layer: projects the input to a higher-dimensional space (width).
    Usually a pointwise 1x1 convolution (Dense layer).
    """
    width: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.width)(x)
