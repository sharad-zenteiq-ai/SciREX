from typing import Optional
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
    projection_hidden_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, in_ch)
        returns: (batch, nx, ny, out_channels)
        """
        # 1. Lifting
        x = Lifting(width=self.width)(x)
        
        # 2. Wavelet Blocks
        for i in range(self.depth):
            # The original paper skips activation in the last block's layer sum
            # before entering the multi-layer projection
            block_activation = self.activation if i < self.depth - 1 else None
            
            x = WaveletBlock2D(
                width=self.width, 
                levels=self.levels,
                wavelet=self.wavelet,
                activation=block_activation
            )(x)
            
        # 3. Projection
        # Paper uses 2nd hidden dim in projection (e.g., width -> 192 -> 1)
        x = Projection(
            out_channels=self.out_channels,
            hidden_dim=self.projection_hidden_dim,
            activation=self.activation
        )(x)
        return x
