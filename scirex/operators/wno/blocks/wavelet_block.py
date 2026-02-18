from typing import Optional
from flax import linen as nn
import jax.numpy as jnp
from ..layers.wavelet_conv import WaveletConv1D, WaveletConv2D

class WaveletBlock1D(nn.Module):
    width: int
    levels: int = 1
    wavelet: str = "haar"
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Wavelet Convolution
        y = WaveletConv1D(
            in_channels=self.width, 
            out_channels=self.width, 
            levels=self.levels,
            wavelet=self.wavelet
        )(x)
        
        # 2. Linear shortcut (Residual connection)
        shortcut = nn.Dense(self.width)(x)
        
        # 3. Sum and Activation
        out = y + shortcut
        if self.activation == "gelu":
            out = nn.gelu(out)
        elif self.activation == "relu":
            out = nn.relu(out)
        return out

class WaveletBlock2D(nn.Module):
    width: int
    levels: int = 1
    wavelet: str = "haar"
    activation: Optional[str] = "gelu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Wavelet Convolution
        y = WaveletConv2D(
            in_channels=self.width, 
            out_channels=self.width, 
            levels=self.levels,
            wavelet=self.wavelet
        )(x)
        
        # 2. Linear shortcut
        shortcut = nn.Dense(self.width)(x)
        
        # 3. Sum and Activation
        out = y + shortcut
        if self.activation == "gelu":
            out = nn.gelu(out)
        elif self.activation == "relu":
            out = nn.relu(out)
        elif self.activation == "mish":
            out = out * jnp.tanh(nn.softplus(out))
        return out
