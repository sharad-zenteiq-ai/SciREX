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

from typing import Tuple, Sequence, Callable, Optional
import jax.numpy as jnp
from flax import linen as nn

from ..layers.fast_wavelet_conv import LiftingWaveletConv2D
from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import GridEmbedding
from ..layers.padding import DomainPadding

class FWNO2D(nn.Module):
    """
    Fast Wavelet Neural Operator (FWNO) 2D.
    
    This model uses the Lifting Scheme (Fast Wavelet Transform) instead of 
    convolution-based DWT for improved efficiency.
    
    Structure:
    1. Grid appending (x, y coordinates)
    2. Lifting (projection to hidden_channels)
    3. Fast Wavelet Layers (Lifting Scheme)
    4. Projection (projection to out_channels)
    """
    hidden_channels: int = 64
    n_layers: int = 4
    out_channels: int = 1
    levels: int = 1
    use_grid: bool = True
    padding: float = 0.0
    activation: Callable = nn.gelu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, in_channels)
        returns: (batch, nx, ny, out_channels)
        """
        original_shape = x.shape
        
        # 1. Grid Embedding
        if self.use_grid:
            x = GridEmbedding(grid_boundaries=((0.0, 1.0), (0.0, 1.0)))(x)
            
        # 2. Domain Padding
        if self.padding > 0:
            pad_layer = DomainPadding(padding=self.padding)
            x = pad_layer(x)

        # 3. Lifting: project to hidden_channels
        x = ChannelMLP(out_channels=self.hidden_channels, n_layers=1, activation=self.activation)(x)
        
        # 4. Fast Wavelet Layers
        for i in range(self.n_layers):
            x = LiftingWaveletConv2D(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                level=self.levels,
                activation=self.activation
            )(x)
            
        # 5. Projection: project to out_channels
        x = ChannelMLP(
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels * 2,
            n_layers=2,
            activation=self.activation
        )(x)
        
        # 6. Inverse Domain Padding (Crop)
        if self.padding > 0:
            x = pad_layer(x, inverse=True, original_shape=original_shape)
            
        return x
