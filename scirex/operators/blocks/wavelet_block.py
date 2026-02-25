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

from typing import Callable
from flax import linen as nn
import jax.numpy as jnp
from ..layers.wavelet_conv import WaveletConv1D, WaveletConv2D

class WaveletBlock1D(nn.Module):
    """
    1D Wavelet Block: WaveletConv1D + Pointwise Skip + Activation.
    """
    hidden_channels: int
    levels: int = 1
    wavelet: str = "haar"
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Wavelet Convolution
        y = WaveletConv1D(
            in_channels=self.hidden_channels, 
            out_channels=self.hidden_channels, 
            levels=self.levels,
            wavelet=self.wavelet
        )(x)
        
        # 2. Linear shortcut (Residual connection)
        shortcut = nn.Dense(self.hidden_channels)(x)
        
        # 3. Sum and Activation
        return self.activation(y + shortcut)

class WaveletBlock2D(nn.Module):
    """
    2D Wavelet Block: WaveletConv2D + Pointwise Skip + Activation.
    """
    hidden_channels: int
    levels: int = 1
    wavelet: str = "haar"
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Wavelet Convolution
        y = WaveletConv2D(
            in_channels=self.hidden_channels, 
            out_channels=self.hidden_channels, 
            levels=self.levels,
            wavelet=self.wavelet
        )(x)
        
        # 2. Linear shortcut
        shortcut = nn.Dense(self.hidden_channels)(x)
        
        # 3. Sum and Activation
        return self.activation(y + shortcut)
