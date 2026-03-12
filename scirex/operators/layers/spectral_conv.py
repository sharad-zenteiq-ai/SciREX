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

from typing import Tuple
from flax import linen as nn
import jax
import jax.numpy as jnp

class SpectralConv(nn.Module):
    """
    N-dimensional Spectral Convolution layer (supports 2D and 3D).
    
    The dimensionality is automatically inferred from the length of ``n_modes``:
      - ``len(n_modes) == 2`` → 2D spectral convolution (RFFT2)
      - ``len(n_modes) == 3`` → 3D spectral convolution (RFFT3)
    
    This layer performs a convolution in the Fourier domain by:
    1. Transforming the input to the frequency domain using a Real Fast Fourier Transform (RFFT).
    2. Multiplying the lower Fourier modes by learnable complex weights.
    3. Inverse transforming the filtered signal back to the spatial domain.
    
    This operation provides a global receptive field, as each Fourier mode 
    carries information from the entire spatial domain.
    
    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_modes (Tuple[int, ...]): Number of Fourier modes to retain for each spatial dimension.
            Length 2 for 2D, length 3 for 3D.
        init_std (float, optional): Standard deviation for weight initialization.
    """
    in_channels: int
    out_channels: int
    n_modes: Tuple[int, ...]
    init_std: float = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: (batch, dim1, dim2, ..., dimN, in_channels)
        n_dim = len(self.n_modes)
        batch = x.shape[0]
        spatial_dims = x.shape[1:-1]
        
        # 1. FFT
        axes = tuple(range(1, n_dim + 1))
        x_ft = jnp.fft.rfftn(x, axes=axes, norm="ortho")
        
        # 2. Weights
        scale = 0.05 if self.init_std is None else self.init_std
        weights_shape = (self.in_channels, self.out_channels) + self.n_modes
        
        # For N dimensions, the number of corners in frequency space is 2**(N-1).
        # We generate combinations of (start, end) for all dimensions except the last spatial one.
        # The last spatial dimension is only positive frequencies (due to RFFT).
        n_corners = 2**(n_dim - 1)
        weights = [
            self.param(f'weights_{i+1}', jax.nn.initializers.normal(stddev=scale), weights_shape, jnp.complex64)
            for i in range(n_corners)
        ]
        
        # Create output tensor in frequency domain
        out_ft_shape = (batch,) + spatial_dims[:-1] + (spatial_dims[-1] // 2 + 1,) + (self.out_channels,)
        out_ft = jnp.zeros(out_ft_shape, dtype=jnp.complex64)
        
        # Build dynamic einsum string (e.g., N=2 -> "bxyi,ioxy->bxyo")
        import itertools
        spatial_letters = "xyzuvw"[:n_dim]
        einsum_str = f"b{spatial_letters}i,io{spatial_letters}->b{spatial_letters}o"
        
        # Iterate over all combinations of positive and negative frequencies for the first N-1 dims.
        # 1 means [:modes], -1 means [-modes:]
        corner_idx = 0
        for signs in itertools.product([1, -1], repeat=n_dim - 1):
            slices_in = [slice(None)]  # batch
            slices_out = [slice(None)] # batch
            
            for d, sign in enumerate(signs):
                modes = self.n_modes[d]
                if sign == 1:
                    slices_in.append(slice(None, list(self.n_modes)[d]))
                    slices_out.append(slice(None, list(self.n_modes)[d]))
                else:
                    slices_in.append(slice(-list(self.n_modes)[d], None))
                    slices_out.append(slice(-list(self.n_modes)[d], None))
                    
            # Last spatial dimension (always positive frequencies for rfft)
            slices_in.append(slice(None, self.n_modes[-1]))
            slices_out.append(slice(None, self.n_modes[-1]))
            
            # Channel dimension
            slices_in.append(slice(None))
            slices_out.append(slice(None))
            
            # Apply einsum for this corner
            x_corner = x_ft[tuple(slices_in)]
            w_corner = weights[corner_idx]
            
            out_corner = jnp.einsum(einsum_str, x_corner, w_corner)
            out_ft = out_ft.at[tuple(slices_out)].set(out_corner)
            
            corner_idx += 1
            
        # 4. Inverse FFT
        x = jnp.fft.irfftn(out_ft, s=spatial_dims, axes=axes, norm="ortho")
        return x

