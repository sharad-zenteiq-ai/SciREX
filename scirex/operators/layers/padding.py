from typing import Tuple, List, Union
from flax import linen as nn
import jax.numpy as jnp

class DomainPadding(nn.Module):
    """
    Applies domain padding to the input tensor.
    This is used to handle non-periodic boundary conditions in FNO.
    
    Parameters
    ----------
    padding : float or list of floats
        The percentage of padding to apply per dimension.
        If a single float, the same padding is applied to all spatial dimensions.
    mode : str
        The padding mode. Default is 'constant' (zero padding).
    """
    padding: Union[float, List[float]]
    mode: str = 'constant'

    def setup(self):
        # Logic for calculating actual pixel padding will happen in __call__
        # to handle dynamic input resolutions.
        pass

    def __call__(self, x: jnp.ndarray, inverse: bool = False, original_shape: Tuple = None) -> jnp.ndarray:
        """
        x: (batch, spatial_1, ..., spatial_n, channels)
        inverse: if True, crops the tensor back to original_shape.
        """
        ndim = x.ndim - 2  # Spatial dimensions
        
        if isinstance(self.padding, (float, int)):
            pad_list = [self.padding] * ndim
        else:
            pad_list = self.padding

        if not inverse:
            # Calculate pixel padding (symmetric)
            pad_width = []
            pad_width.append((0, 0)) # Batch
            for i in range(ndim):
                p = int(round(x.shape[i+1] * pad_list[i]))
                pad_width.append((p, p)) # Symmetric padding
            pad_width.append((0, 0)) # Channels
            
            return jnp.pad(x, pad_width, mode=self.mode)
        else:
            # Crop back
            if original_shape is None:
                raise ValueError("original_shape must be provided for inverse padding (cropping)")
            
            slices = [slice(None)] # Batch
            for i in range(ndim):
                p = int(round(original_shape[i+1] * pad_list[i]))
                slices.append(slice(p, p + original_shape[i+1]))
            slices.append(slice(None)) # Channels
            
            return x[tuple(slices)]
