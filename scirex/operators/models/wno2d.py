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

from typing import Optional, Callable
from flax import linen as nn
import jax.numpy as jnp
from ..layers.lifting import Lifting
from ..layers.projection import Projection
from ..blocks.wavelet_block import WaveletBlock2D

class WNO2D(nn.Module):
    """
    2D Wavelet Neural Operator model.
    Structure: Lifting -> n_layers x WaveletBlock -> Projection
    """
    hidden_channels: int
    n_layers: int
    levels: int = 1
    wavelet: str = "haar"
    out_channels: int = 1
    activation: Callable = nn.gelu
    projection_hidden_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, in_channels)
        returns: (batch, nx, ny, out_channels)
        """
        # 1. Lifting
        x = Lifting(hidden_channels=self.hidden_channels)(x)
        
        # 2. Wavelet Blocks
        for i in range(self.n_layers):
            # The original paper skips activation in the last block's layer sum
            # before entering the multi-layer projection
            block_activation = self.activation if i < self.n_layers - 1 else (lambda x: x)
            
            x = WaveletBlock2D(
                hidden_channels=self.hidden_channels, 
                levels=self.levels,
                wavelet=self.wavelet,
                activation=block_activation
            )(x)
            
        # 3. Projection
        # Paper uses 2nd hidden dim in projection (e.g., width -> 192 -> 1)
        x = Projection(
            out_channels=self.out_channels,
            hidden_dim=self.projection_hidden_dim,
            activation=self.activation
        )(x)
        return x
