from flax import linen as nn
import jax.numpy as jnp
from ..layers.spectral_conv import SpectralConv2D

class SpectralBlock(nn.Module):
    """
    Standard FNO Block: SpectralConv2D + Pointwise 1x1 Conv + Residual Connection + Activation.
    """
    width: int
    modes_x: int
    modes_y: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Spectral convolution
        y_s = SpectralConv2D(
            in_channels=self.width, 
            out_channels=self.width, 
            modes_x=self.modes_x, 
            modes_y=self.modes_y
        )(x)
        
        # Pointwise 1x1 convolution
        y_p = nn.Dense(self.width)(x)
        
        # Residual connection + activation (GeLU)
        return nn.gelu(x + y_s + y_p)
