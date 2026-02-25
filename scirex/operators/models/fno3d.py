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

from typing import Optional, List, Union, Tuple
from flax import linen as nn
import jax.numpy as jnp
from ..layers.lifting import Lifting
from ..layers.projection import Projection
from ..layers.padding import DomainPadding
from ..blocks.fno_block import SpectralBlock3D

class FNO3D(nn.Module):
    """
    Refined 3D Fourier Neural Operator model.
    Includes Domain Padding and Instance Normalization for improved stability.
    """
    hidden_channels: int
    n_layers: int
    n_modes: Tuple[int, int, int]
    out_channels: int
    projection_hidden_dim: int = 128
    padding: Union[float, List[float]] = 0.0
    use_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, nz, in_channels)
        """
        original_shape = x.shape
        
        # 1. Domain Padding (to handle non-periodic conditions)
        if self.padding > 0:
            pad_layer = DomainPadding(padding=self.padding)
            x = pad_layer(x)
            
        # 2. Lifting: (batch, nx_p, ny_p, nz_p, in_ch) -> (batch, ..., hidden_channels)
        x = Lifting(hidden_channels=self.hidden_channels)(x)
        
        # 3. Iterative FNO blocks
        for _ in range(self.n_layers):
            x = SpectralBlock3D(
                hidden_channels=self.hidden_channels, 
                n_modes=self.n_modes,
                use_norm=self.use_norm
            )(x)
            
        # 4. Projection
        x = Projection(
            out_channels=self.out_channels, 
            hidden_dim=self.projection_hidden_dim
        )(x)
        
        # 5. Inverse Domain Padding (Crop)
        if self.padding > 0:
            x = pad_layer(x, inverse=True, original_shape=original_shape)
            
        return x
