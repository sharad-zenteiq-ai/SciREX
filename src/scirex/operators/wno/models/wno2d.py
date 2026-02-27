from typing import Optional, List, Tuple, Union
from flax import linen as nn
import jax.numpy as jnp
from ..blocks.wavelet_block import WaveletBlock2D


class WNO2D(nn.Module):
    """
    2D Wavelet Neural Operator model following the reference implementation
    (TapasTripura/WNO, Version 2.0.0).
    
    Architecture:
    1. Lift the input using v(x,y) = fc0(a(x,y), x, y).
       Grid coordinates (x, y) are appended to the input before lifting.
    2. L layers of wavelet integral operators:
       v(j+1)(x,y) = sigma(K.v + W.v)(x,y)
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
    size : tuple of int
        Signal dimensions [height, width] at training resolution.
    wavelet : str
        Wavelet filter name (default: 'db6' as in reference Darcy example).
    in_channel : int
        Number of input channels including appended grid coordinates.
    out_channels : int
        Number of output channels.
    grid_range : list of float
        Right supports of 2D domain [x_max, y_max] for grid construction.
    padding : int
        Size of zero-padding for non-periodic boundaries.
    activation : str
        Activation for intermediate wavelet layers (default: 'mish').
    """
    width: int = 64
    depth: int = 4
    level: int = 3
    size: Tuple[int, int] = (64, 64)
    wavelet: str = "db6"
    in_channel: int = 3
    out_channels: int = 1
    grid_range: Tuple[float, float] = (1.0, 1.0)
    padding: int = 0
    activation: str = "mish"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (batch, nx, ny, in_features).
               For Darcy: in_features=1, i.e., a(x,y) only.
               Grid coordinates (x, y) will be appended automatically.
        Returns:
            Output tensor of shape (batch, nx, ny, out_channels).
        """
        # 0. Append grid coordinates (matching reference: torch.cat((x, grid), dim=-1))
        grid = self._get_grid(x.shape)
        x = jnp.concatenate([x, grid], axis=-1)
        
        # 1. Lifting: fc0 (Linear(in_channel, width))
        x = nn.Dense(self.width)(x)  # Shape: Batch * H * W * width
        
        # 2. Padding (if non-periodic)
        if self.padding != 0:
            x = jnp.pad(x, ((0, 0), (0, self.padding), (0, self.padding), (0, 0)))
        
        # 3. Wavelet integral layers with residual connections
        for i in range(self.depth):
            # Last layer has no activation (matching reference)
            block_activation = self.activation if i < self.depth - 1 else None
            
            x = WaveletBlock2D(
                width=self.width, 
                level=self.level,
                size=self.size,
                wavelet=self.wavelet,
                activation=block_activation
            )(x)
        
        # 4. Remove padding
        if self.padding != 0:
            x = x[:, :-self.padding, :-self.padding, :]
        
        # 5. Projection: fc1(width -> 128) + GELU + fc2(128 -> out_channels)
        # Matches reference: nn.Linear(width, 128) + F.gelu + nn.Linear(128, 1)
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_channels)(x)
        return x
    
    def _get_grid(self, shape):
        """Constructs a 2D grid of coordinates matching the reference."""
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = jnp.linspace(0, self.grid_range[0], size_x)
        gridx = gridx.reshape(1, size_x, 1, 1)
        gridx = jnp.broadcast_to(gridx, (batchsize, size_x, size_y, 1))
        
        gridy = jnp.linspace(0, self.grid_range[1], size_y)
        gridy = gridy.reshape(1, 1, size_y, 1)
        gridy = jnp.broadcast_to(gridy, (batchsize, size_x, size_y, 1))
        
        return jnp.concatenate([gridx, gridy], axis=-1)
