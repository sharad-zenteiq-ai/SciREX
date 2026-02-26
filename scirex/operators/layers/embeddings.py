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
import jax.numpy as jnp

class GridEmbedding(nn.Module):
    """
    Appends normalized grid coordinates to the input tensor.
    Equivalent to neuraloperator's GridEmbeddingND.
    
    For a 2D input (batch, nx, ny, c), it appends 2 channels (x, y).
    For a 3D input (batch, nx, ny, nz, c), it appends 3 channels (x, y, z).
    """
    grid_boundaries: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))

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
            coords = jnp.linspace(start, end, res)
            # Reshape it to be broadcastable to the full spatial shape
            # E.g. for 2D (nx, ny), x-coords: (nx, 1), y-coords: (1, ny)
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
