from flax import linen as nn
import jax.numpy as jnp
from ...layers.lifting import Lifting
from ...layers.projection import Projection
from ..blocks.wavelet_block import WaveletBlock1D

class WNO1D(nn.Module):
    """
    1D Wavelet Neural Operator model.
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
        x: (batch, length, in_ch)
        returns: (batch, length, out_channels)
        """
        # 1. Lifting
        x = Lifting(width=self.width)(x)
        
        # 2. Wavelet Blocks
        for _ in range(self.depth):
            x = WaveletBlock1D(
                width=self.width, 
                levels=self.levels,
                wavelet=self.wavelet,
                activation=self.activation
            )(x)
            
        # 3. Projection
        x = Projection(out_channels=self.out_channels)(x)
        return x
