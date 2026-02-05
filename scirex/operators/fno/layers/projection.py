from flax import linen as nn
import jax.numpy as jnp

class Projection(nn.Module):
    """
    Projection layer: projects the hidden representation to the desired output channels.
    Usually a pointwise 1x1 convolution (Dense layer).
    """
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.out_channels)(x)
