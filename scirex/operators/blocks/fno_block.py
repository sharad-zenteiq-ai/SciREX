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

from typing import Optional, Callable, Tuple
from flax import linen as nn
import jax.numpy as jnp
from ..layers.spectral_conv import SpectralConv2D, SpectralConv3D

class SpectralBlock(nn.Module):
    """
    Refined FNO Block: SpectralConv2D + Pointwise Skip + Normalization + Activation.
    """
    hidden_channels: int
    n_modes: Tuple[int, int]
    activation: Callable = nn.gelu
    use_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Spectral branch
        y_s = SpectralConv2D(
            in_channels=self.hidden_channels, 
            out_channels=self.hidden_channels, 
            n_modes=self.n_modes
        )(x)
        
        # 2. Pointwise Skip branch
        y_p = nn.Dense(self.hidden_channels)(x)
        
        # 3. Combine
        x = y_s + y_p
        
        # 4. Normalization (InstanceNorm is standard for FNO)
        if self.use_norm:
            x = nn.InstanceNorm()(x)
        
        # 5. Activation
        return self.activation(x)

class SpectralBlock3D(nn.Module):
    """
    Refined 3D FNO Block: SpectralConv3D + Pointwise Skip + Normalization + Activation.
    """
    hidden_channels: int
    n_modes: Tuple[int, int, int]
    activation: Callable = nn.gelu
    use_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Spectral branch
        y_s = SpectralConv3D(
            in_channels=self.hidden_channels, 
            out_channels=self.hidden_channels, 
            n_modes=self.n_modes
        )(x)
        
        # 2. Pointwise Skip branch
        y_p = nn.Dense(self.hidden_channels)(x)
        
        # 3. Combine
        x = y_s + y_p
        
        # 4. Normalization
        if self.use_norm:
            x = nn.InstanceNorm()(x)
            
        # 5. Activation
        return self.activation(x)
