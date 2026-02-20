from typing import Optional, Callable
from flax import linen as nn
import jax.numpy as jnp
from ..layers.spectral_conv import SpectralConv2D, SpectralConv3D

class SpectralBlock(nn.Module):
    """
    Refined FNO Block: SpectralConv2D + Pointwise Skip + Normalization + Activation.
    """
    width: int
    modes_x: int
    modes_y: int
    activation: Callable = nn.gelu
    use_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Spectral branch
        y_s = SpectralConv2D(
            in_channels=self.width, 
            out_channels=self.width, 
            modes_x=self.modes_x, 
            modes_y=self.modes_y
        )(x)
        
        # 2. Pointwise Skip branch
        y_p = nn.Dense(self.width)(x)
        
        # 3. Combine
        x = y_s + y_p
        
        # 4. Normalization (InstanceNorm is standard for FNO)
        if self.use_norm:
            x = nn.InstanceNorm()(x)
        
        # 5. Activation
        return self.activation(x)

class SpectralBlock3D(nn.Module):
    """
    Refined 3D FNO Block: SpectralConv3D + Pointwise Skip + Normalization + Activation.
    """
    width: int
    modes_x: int
    modes_y: int
    modes_z: int
    activation: Callable = nn.gelu
    use_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Spectral branch
        y_s = SpectralConv3D(
            in_channels=self.width, 
            out_channels=self.width, 
            modes_x=self.modes_x, 
            modes_y=self.modes_y,
            modes_z=self.modes_z
        )(x)
        
        # 2. Pointwise Skip branch
        y_p = nn.Dense(self.width)(x)
        
        # 3. Combine
        x = y_s + y_p
        
        # 4. Normalization
        if self.use_norm:
            x = nn.InstanceNorm()(x)
            
        # 5. Activation
        return self.activation(x)
