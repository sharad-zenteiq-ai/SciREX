from typing import Optional, List
from flax import linen as nn
import jax.numpy as jnp
from ..blocks.wavelet_block import WaveletBlock1D


class WNO1D(nn.Module):
    """
    1D Wavelet Neural Operator model following the reference implementation
    (TapasTripura/WNO, Version 2.0.0).
    
    Architecture:
    1. Lift the input using v(x) = fc0(a(x), x).
       Grid coordinates are appended to the input before lifting.
    2. L layers of wavelet integral operators:
       v(j+1)(x) = sigma(K.v + W.v)(x)
       where K is wavelet conv, W is 1x1 conv, sigma is mish (except last layer).
    3. Project output: fc1(width -> 128) + GELU + fc2(128 -> out_channels).
    
    Parameters
    ----------
    width : int
        Lifting dimension (hidden width of the model).
    depth : int
        Number of wavelet kernel integral layers.
    level : int
        Number of wavelet decomposition levels.
    size : int
        Signal length at training resolution.
    wavelet : str
        Wavelet filter name (default: 'db6' as in reference Burgers example).
    in_channel : int
        Number of input channels including appended grid coordinate.
    out_channels : int
        Number of output channels.
    grid_range : float
        Right support of the 1D domain for grid construction.
    padding : int
        Size of zero-padding for non-periodic boundaries.
    activation : str
        Activation for intermediate wavelet layers (default: 'mish').
    """
    width: int = 64
    depth: int = 4
    level: int = 8
    size: int = 1024
    wavelet: str = "db6"
    in_channel: int = 2
    out_channels: int = 1
    grid_range: float = 1.0
    padding: int = 0
    activation: str = "mish"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (batch, length, in_features).
               For Burgers: in_features=1, i.e., a(x) only.
               Grid coordinate x will be appended automatically.
        Returns:
            Output tensor of shape (batch, length, out_channels).
        """
        # 0. Append grid coordinates (matching reference: torch.cat((x, grid), dim=-1))
        grid = self._get_grid(x.shape)
        x = jnp.concatenate([x, grid], axis=-1)
        
        # 1. Lifting: fc0 (Linear(in_channel, width))
        x = nn.Dense(self.width)(x)  # Shape: Batch * L * width
        
        # 2. Padding (if non-periodic)
        if self.padding != 0:
            x = jnp.pad(x, ((0, 0), (0, self.padding), (0, 0)))
        
        # 3. Wavelet integral layers with residual connections
        for i in range(self.depth):
            # Last layer has no activation (matching reference)
            block_activation = self.activation if i < self.depth - 1 else None
            
            x = WaveletBlock1D(
                width=self.width, 
                level=self.level,
                size=self.size,
                wavelet=self.wavelet,
                activation=block_activation
            )(x)
        
        # 4. Remove padding
        if self.padding != 0:
            x = x[:, :-self.padding, :]
        
        # 5. Projection: fc1(width -> 128) + GELU + fc2(128 -> out_channels)
        # Matches reference: nn.Linear(width, 128) + F.gelu + nn.Linear(128, 1)
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_channels)(x)
        return x
    
    def _get_grid(self, shape):
        """Constructs a 1D grid of coordinates matching the reference."""
        batchsize, size_x = shape[0], shape[1]
        gridx = jnp.linspace(0, self.grid_range, size_x)
        gridx = gridx.reshape(1, size_x, 1)
        gridx = jnp.broadcast_to(gridx, (batchsize, size_x, 1))
        return gridx
