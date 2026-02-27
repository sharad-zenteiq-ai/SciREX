from typing import Tuple, Sequence
import jax.numpy as jnp
from flax import linen as nn

from scirex.operators.wno.layers.fast_wavelet_conv import LiftingWaveletConv2D
from scirex.operators.layers.lifting import Lifting
from scirex.operators.layers.projection import Projection

class FWNO2D(nn.Module):
    """
    Fast Wavelet Neural Operator (FWNO) 2D.
    
    This model uses the Lifting Scheme (Fast Wavelet Transform) instead of 
    convolution-based DWT for improved efficiency.
    
    Structure:
    1. Grid appending (x, y coordinates)
    2. Lifting (projection to width)
    3. Fast Wavelet Layers (Lifting Scheme)
    4. Projection (projection to out_channels)
    """
    width: int = 64
    depth: int = 4
    channels: int = 1
    out_channels: int = 1
    level: int = 1 # Not used in Lifting Scheme directly (we do single level processing per layer)
    activation: str = "gelu"
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, nx, ny, channels)
        
        # 1. Append Grid Coordinates
        batch_size, nx, ny, _ = x.shape
        x_grid = jnp.linspace(0, 1, nx)
        y_grid = jnp.linspace(0, 1, ny)
        X, Y = jnp.meshgrid(x_grid, y_grid, indexing='ij')
        grid = jnp.stack([X, Y], axis=-1) # (nx, ny, 2)
        grid = jnp.tile(grid[None, ...], (batch_size, 1, 1, 1))
        
        x = jnp.concatenate([x, grid], axis=-1)
        
        # 2. Lifting (Input Projection)
        x = nn.Dense(self.width)(x)
        x = jnp.transpose(x, (0, 1, 2, 3)) # Ensure layout is correct
        
        # 3. Fast Wavelet Layers
        # We process 'depth' number of layers
        # Each layer applies the Lifting Scheme Transform -> Weighting -> Inverse Lifting
        
        for i in range(self.depth):
            x = LiftingWaveletConv2D(
                in_channels=self.width,
                out_channels=self.width,
                level=self.level
            )(x)
            
            # Activation is inside the layer (at the end), but we can add more if needed
            # The layer ends with activation(out + res).
            
        # 4. Projection (Output Projection)
        # Standard MLP: width -> 128 -> out_channels
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_channels)(x)
        
        return x
