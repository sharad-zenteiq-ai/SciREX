from typing import Optional, List, Union
from flax import linen as nn
import jax.numpy as jnp
from ...layers.lifting import Lifting
from ...layers.projection import Projection
from ...layers.padding import DomainPadding
from ..blocks.fno_block import SpectralBlock3D

class FNO3D(nn.Module):
    """
    Refined 3D Fourier Neural Operator model.
    Includes Domain Padding and Instance Normalization for improved stability.
    """
    width: int
    depth: int
    modes_x: int
    modes_y: int
    modes_z: int
    out_channels: int
    projection_hidden_dim: int = 128
    padding: Union[float, List[float]] = 0.0
    use_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, nz, in_ch)
        """
        original_shape = x.shape
        
        # 1. Domain Padding (to handle non-periodic conditions)
        if self.padding > 0:
            pad_layer = DomainPadding(padding=self.padding)
            x = pad_layer(x)
            
        # 2. Lifting: (batch, nx_p, ny_p, nz_p, in_ch) -> (batch, ..., width)
        x = Lifting(width=self.width)(x)
        
        # 3. Iterative FNO blocks
        for _ in range(self.depth):
            x = SpectralBlock3D(
                width=self.width, 
                modes_x=self.modes_x, 
                modes_y=self.modes_y,
                modes_z=self.modes_z,
                use_norm=self.use_norm
            )(x)
            
        # 4. Projection
        x = Projection(
            out_channels=self.out_channels, 
            hidden_dim=self.projection_hidden_dim
        )(x)
        
        # 5. Inverse Domain Padding (Crop)
        if self.padding > 0:
            x = pad_layer(x, inverse=True, original_shape=original_shape)
            
        return x
