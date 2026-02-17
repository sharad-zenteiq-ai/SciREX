from flax import linen as nn
import jax.numpy as jnp
from ...layers.lifting import Lifting
from ...layers.projection import Projection
from ..blocks.wavelet_block import WaveletBlock2D

class WNO2D(nn.Module):
    """
    2D Wavelet Neural Operator model.
    Structure: Lifting -> n x WaveletBlock -> Projection
    """
    width: int
    depth: int
    levels: int = 1
    wavelet: str = "haar"
    out_channels: int = 1
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, in_ch)
        returns: (batch, nx, ny, out_channels)
        """
        # 1. Lifting
        x = Lifting(width=self.width)(x)
        
        # 2. Wavelet Blocks
        for _ in range(self.depth):
            x = WaveletBlock2D(
                width=self.width, 
                levels=self.levels,
                wavelet=self.wavelet,
                activation=self.activation
            )(x)
            
        # 3. Projection
        x = Projection(out_channels=self.out_channels)(x)
        return x
