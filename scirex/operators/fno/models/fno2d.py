from flax import linen as nn
import jax.numpy as jnp
from ..layers.lifting import Lifting
from ..layers.projection import Projection
from ..blocks.fno_block import SpectralBlock

class FNO2D(nn.Module):
    """
    2D Fourier Neural Operator model.
    Structure: Lifting -> n x SpectralBlock -> Projection
    """
    width: int
    depth: int
    modes_x: int
    modes_y: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, in_ch)
        returns: (batch, nx, ny, out_channels)
        """
        # Lifting: encoder Dense -> project to depth width
        x = Lifting(width=self.width)(x)
        
        # Iterative FNO blocks
        for _ in range(self.depth):
            x = SpectralBlock(
                width=self.width, 
                modes_x=self.modes_x, 
                modes_y=self.modes_y
            )(x)
            
        # Projection: decoder Dense -> project to desired output channels
        x = Projection(out_channels=self.out_channels)(x)
        return x
