# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

from typing import Tuple, List, Union
from flax import linen as nn
import jax.numpy as jnp

class DomainPadding(nn.Module):
    """
    Symmetric Spatial Domain Padding.
    
    FNO models assume periodic boundary conditions due to the underlying FFT.
    DomainPadding mitigates boundary artifacts when dealing with non-periodic 
    PDEs by expanding the spatial domain before processing and cropping 
    the result back to the original resolution.

    Args:
        padding (float or List[float]): Percentage of padding (0.0 to 1.0) 
            to apply per spatial dimension.
        mode (str): JAX padding mode (default: 'constant' for zero-padding).
    """
    padding: Union[float, List[float]]
    mode: str = 'constant'
    
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
            if len(pad_list) != ndim:
                raise ValueError(f"Length of padding list ({len(pad_list)}) must match number of spatial dimensions ({ndim}).")

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
