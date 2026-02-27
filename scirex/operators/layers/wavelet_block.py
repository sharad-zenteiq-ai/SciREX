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

from typing import Optional, Union, Tuple, Callable
from flax import linen as nn
import jax.numpy as jnp
from .wavelet_conv import WaveletConv1D, WaveletConv2D


def mish(x):
    """Mish activation: x * tanh(softplus(x)). 
    Matches F.mish() used in the reference WNO implementation."""
    return x * jnp.tanh(nn.softplus(x))


class WaveletBlock1D(nn.Module):
    """
    Single WNO integral block for 1D problems.
    
    Implements: v(j+1)(x) = sigma(K.v + W.v)(x)
    where K is the wavelet convolution and W is a 1x1 (Dense) linear shortcut.
    
    Matches the reference WNO (TapasTripura/WNO):
    - Uses mish activation by default (intermediate layers).
    - Last layer should have activation=None (no activation).
    """
    hidden_channels: int
    levels: int = 1
    size: int = 1024
    wavelet: str = "db4"
    activation: Union[str, Callable, None] = "mish"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Wavelet Convolution (K.v)
        y = WaveletConv1D(
            in_channels=self.hidden_channels, 
            out_channels=self.hidden_channels, 
            levels=self.levels,
            wavelet=self.wavelet
        )(x)
        
        # 2. Linear shortcut W.v (1x1 conv / pointwise linear)
        shortcut = nn.Dense(self.hidden_channels)(x)
        
        # 3. Sum and Activation: sigma(K.v + W.v)
        out = y + shortcut
        
        if self.activation is None:
            return out
            
        if callable(self.activation):
            return self.activation(out)
            
        if self.activation == "mish":
            return mish(out)
        elif self.activation == "gelu":
            return nn.gelu(out)
        elif self.activation == "relu":
            return nn.relu(out)
            
        return out


class WaveletBlock2D(nn.Module):
    """
    Single WNO integral block for 2D problems.
    
    Implements: v(j+1)(x,y) = sigma(K.v + W.v)(x,y)
    where K is the wavelet convolution and W is a 1x1 (Dense) linear shortcut.
    
    Matches the reference WNO (TapasTripura/WNO):
    - Uses mish activation by default (intermediate layers).
    - Last layer should have activation=None (no activation).
    """
    hidden_channels: int
    levels: int = 1
    size: Tuple[int, int] = (64, 64)
    wavelet: str = "db4"
    activation: Union[str, Callable, None] = "mish"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Wavelet Convolution (K.v)
        y = WaveletConv2D(
            in_channels=self.hidden_channels, 
            out_channels=self.hidden_channels, 
            levels=self.levels,
            wavelet=self.wavelet
        )(x)
        
        # 2. Linear shortcut W.v (1x1 conv / pointwise linear)
        shortcut = nn.Dense(self.hidden_channels)(x)
        
        # 3. Sum and Activation: sigma(K.v + W.v)
        out = y + shortcut
        
        if self.activation is None:
            return out
            
        if callable(self.activation):
            return self.activation(out)
            
        if self.activation == "mish":
            return mish(out)
        elif self.activation == "gelu":
            return nn.gelu(out)
        elif self.activation == "relu":
            return nn.relu(out)
            
        return out
