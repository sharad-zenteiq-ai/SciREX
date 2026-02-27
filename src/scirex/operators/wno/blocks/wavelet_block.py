from typing import Optional
from flax import linen as nn
import jax.numpy as jnp
from ..layers.wavelet_conv import WaveletConv1D, WaveletConv2D


def mish(x):
    """Mish activation: x * tanh(softplus(x)). 
    Matches F.mish() used in the reference WNO implementation."""
    return x * jnp.tanh(nn.softplus(x))


class WaveletBlock1D(nn.Module):
    """
    Single WNO integral block for 1D problems.
    
    Implements: v(j+1)(x) = sigma(K.v + W.v)(x)
    where K is the wavelet convolution and W is a 1x1 (Dense) linear shortcut.
    
    Matches the reference WNO (TapasTripura/WNO):
    - Uses mish activation by default (intermediate layers).
    - Last layer should have activation=None (no activation).
    """
    width: int
    level: int = 1
    size: int = 1024
    wavelet: str = "db4"
    activation: Optional[str] = "mish"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Wavelet Convolution (K.v)
        y = WaveletConv1D(
            in_channels=self.width, 
            out_channels=self.width, 
            level=self.level,
            size=self.size,
            wavelet=self.wavelet
        )(x)
        
        # 2. Linear shortcut W.v (1x1 conv / pointwise linear)
        shortcut = nn.Dense(self.width)(x)
        
        # 3. Sum and Activation: sigma(K.v + W.v)
        out = y + shortcut
        if self.activation == "mish":
            out = mish(out)
        elif self.activation == "gelu":
            out = nn.gelu(out)
        elif self.activation == "relu":
            out = nn.relu(out)
        # activation=None means no activation (used for last layer)
        return out


class WaveletBlock2D(nn.Module):
    """
    Single WNO integral block for 2D problems.
    
    Implements: v(j+1)(x,y) = sigma(K.v + W.v)(x,y)
    where K is the wavelet convolution and W is a 1x1 (Dense) linear shortcut.
    
    Matches the reference WNO (TapasTripura/WNO):
    - Uses mish activation by default (intermediate layers).
    - Last layer should have activation=None (no activation).
    """
    width: int
    level: int = 1
    size: tuple = (64, 64)
    wavelet: str = "db4"
    activation: Optional[str] = "mish"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Wavelet Convolution (K.v)
        y = WaveletConv2D(
            in_channels=self.width, 
            out_channels=self.width, 
            level=self.level,
            size=self.size,
            wavelet=self.wavelet
        )(x)
        
        # 2. Linear shortcut W.v (1x1 conv / pointwise linear)
        shortcut = nn.Dense(self.width)(x)
        
        # 3. Sum and Activation: sigma(K.v + W.v)
        out = y + shortcut
        if self.activation == "mish":
            out = mish(out)
        elif self.activation == "gelu":
            out = nn.gelu(out)
        elif self.activation == "relu":
            out = nn.relu(out)
        # activation=None means no activation (used for last layer)
        return out
