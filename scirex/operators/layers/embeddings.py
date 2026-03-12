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

from typing import Tuple, Sequence, Optional, Union
from flax import linen as nn
import jax.numpy as jnp

class GridEmbedding(nn.Module):
    """
    Cartesian Coordinate Embedding (Grid Features).
    
    Appends normalized spatial coordinates (x, y, z, ...) as additional 
    input channels. This breaks translation invariance and provides the 
    neural operator with explicit geometric context.

    Attributes:
        grid_boundaries (Tuple): Start and end points for each spatial dimension.
        endpoint (bool): Whether to include the stop point in the grid.
                        Set to False for periodic domains (half-open interval [start, end)).
                        Set to True for non-periodic domains (closed interval [start, end]).
    """
    grid_boundaries: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
    endpoint: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, ..., in_channels)
        returns: (batch, ..., in_channels + n_dim)
        """
        shape = x.shape
        batch_size = shape[0]
        spatial_dims = shape[1:-1]
        n_dim = len(spatial_dims)
        
        # Create grid for each dimension
        grid_coords = []
        for i, (start, end) in enumerate(self.grid_boundaries):
            res = spatial_dims[i]
            # Create a 1D coordinate array
            coords = jnp.linspace(start, end, res, endpoint=self.endpoint)
            
            # Reshape it to be broadcastable to the full spatial shape
            reshape_shape = [1] * n_dim
            reshape_shape[i] = res
            coords = coords.reshape(reshape_shape)
            
            # Broadcast to the full spatial shape
            coords = jnp.broadcast_to(coords, spatial_dims)
            grid_coords.append(coords)
            
        # Stack coordinates: (nx, ny, ..., n_dim)
        grid = jnp.stack(grid_coords, axis=-1)
        
        # Add batch dimension: (batch, nx, ny, ..., n_dim)
        grid = jnp.broadcast_to(grid[None, ...], (batch_size, *spatial_dims, n_dim))
        
        # Concatenate along the channel dimension
        return jnp.concatenate([x, grid], axis=-1)

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal multi-scale embedding (e.g., for NeRF-style or Transformers).
    
    Attributes:
        num_frequencies (int): Number of frequency scales.
        embedding_type (str): 'transformer' or 'nerf'.
        max_positions (int): Base for the frequency scaling in 'transformer' type.
    """
    num_frequencies: int
    embedding_type: str = "transformer"
    max_positions: int = 10000

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, ..., in_channels)
        returns: (batch, ..., in_channels * num_frequencies * 2)
        """
        in_channels = x.shape[-1]
        
        if self.embedding_type == "nerf":
            freqs = (2 ** jnp.arange(self.num_frequencies)) * jnp.pi
        else:
            k = jnp.arange(self.num_frequencies) / self.num_frequencies * 2.0
            freqs = (1.0 / self.max_positions) ** k

        # Compute frequencies: (..., in_channels, num_freqs)
        # Using einsum for flexibility across any number of spatial dimensions
        # x is (..., in_channels), freqs is (num_freqs)
        out = jnp.einsum("...i,j->...ij", x, freqs)
        
        # Stack sin/cos and flatten to channel dimension
        out = jnp.stack((jnp.sin(out), jnp.cos(out)), axis=-1)
        
        # Reshape to (batch, ..., 2 * in_channels * num_frequencies)
        new_shape = x.shape[:-1] + (-1,)
        return out.reshape(new_shape)

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for Channels-Last JAX/Flax.
    
    Attributes:
        dim (int): Dimensionality of the embedding (must be even).
        scale (float): Scaling factor for coordinates.
    """
    dim: int
    scale: float = 1.0

    def setup(self):
        # inv_freq matches the standard RoPE implementation
        self.inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))

    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        coords: (batch, ..., spatial_dim) 
        returns: frequencies (batch, ..., dim)
        """
        # coords are mapped to frequencies
        # coords: (..., C_in), inv_freq: (dim/2,) -> out: (..., C_in, dim/2)
        freqs = jnp.einsum("...i,j->...ij", coords * self.scale, self.inv_freq)
        
        # Repeat to match dim: (batch, ..., C_in, dim)
        freqs = jnp.concatenate((freqs, freqs), axis=-1)
        
        # Flatten the last two dimensions to be the new "channel"
        new_shape = coords.shape[:-1] + (-1,)
        return freqs.reshape(new_shape)

def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotates half the hidden dims of the input."""
    x = x.reshape(x.shape[:-1] + (2, -1))
    x1, x2 = x[..., 0, :], x[..., 1, :]
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(t: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    """Applies RoPE to a tensor."""
    return (t * jnp.cos(freqs)) + (rotate_half(t) * jnp.sin(freqs))