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
        # 1. FFT
        axes = tuple(range(1, n_dim + 1))
        # Use norm="forward" for parity with neuraloperator
        x_ft = jnp.fft.rfftn(x, axes=axes, norm="forward")
        
        # 2. Initialization scale - Aligned with neuraloperator Xavier-style
        if self.init_std is None:
            scale = (2.0 / (self.in_channels + self.out_channels)) ** 0.5
        else:
            scale = self.init_std
            
        # For complex weights, to get total variance of scale^2, 
        # each component's stddev should be scale / sqrt(2)
        complex_scale = scale / (2.0**0.5)

        # 3. Weights
        # Non-RFFT dimensions (all except the last one)
        spatial_dims_ft = list(spatial_dims)
        spatial_dims_ft[-1] = spatial_dims_ft[-1] // 2 + 1
        
        weights_shape = (self.in_channels, self.out_channels) + self.n_modes
        # RFFT dimension (last one) has modes[-1] // 2 + 1 modes if we were filtering there,
        # but n_modes already refers to active modes.
        
        # We need to handle the indexing carefully to match fftshift behavior in N-dims.
        # For simplicity and parity, we restore the 2D/3D specific implementations 
        # alongside the generic one, or refine the generic one.
        # Let's refine the generic one using the "corners" approach but with correct norm and scale.
        
        n_corners = 2**(n_dim - 1)
        weights = [
            self.param(f'weights_{i+1}', jax.nn.initializers.normal(stddev=complex_scale), weights_shape, jnp.complex64)
            for i in range(n_corners)
        ]
        
        # Create output tensor in frequency domain
        out_ft_shape = (batch,) + tuple(spatial_dims_ft) + (self.out_channels,)
        out_ft = jnp.zeros(out_ft_shape, dtype=jnp.complex64)
        
        # Build dynamic einsum string (e.g., N=2 -> "bxyi,ioxy->bxyo")
        import itertools
        spatial_letters = "xyzuvw"[:n_dim]
        einsum_str = f"b{spatial_letters}i,io{spatial_letters}->b{spatial_letters}o"
        
        corner_idx = 0
        for signs in itertools.product([1, -1], repeat=n_dim - 1):
            slices_in = [slice(None)]  # batch
            slices_out = [slice(None)] # batch
            
            for d, sign in enumerate(signs):
                modes = self.n_modes[d]
                if sign == 1:
                    slices_in.append(slice(None, modes))
                    slices_out.append(slice(None, modes))
                else:
                    slices_in.append(slice(-modes, None))
                    slices_out.append(slice(-modes, None))
                    
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
        x = jnp.fft.irfftn(out_ft, s=spatial_dims, axes=axes, norm="forward")

        # 5. Add learned bias
        bias = self.param('bias', jax.nn.initializers.zeros, (self.out_channels,))
        # Expand bias to match (1, 1, ..., 1, channels)
        bias_shape = (1,) * n_dim + (self.out_channels,)
        x = x + bias.reshape(bias_shape)

        return x

class SpectralConv2D(nn.Module):
    """Backward compatibility for 2D spectral convolution."""
    in_channels: int
    out_channels: int
    n_modes: Tuple[int, int]
    init_std: float = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return SpectralConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_modes=self.n_modes,
            init_std=self.init_std
        )(x)

class SpectralConv3D(nn.Module):
    """Backward compatibility for 3D spectral convolution."""
    in_channels: int
    out_channels: int
    n_modes: Tuple[int, int, int]
    init_std: float = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return SpectralConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_modes=self.n_modes,
            init_std=self.init_std
        )(x)
        return x

