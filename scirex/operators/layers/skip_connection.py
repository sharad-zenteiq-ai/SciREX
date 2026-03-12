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

from typing import Literal, Optional
from flax import linen as nn
import jax.numpy as jnp

class SoftGating(nn.Module):
    """
    Parametric channel-wise gating mechanism.
    
    Multiplies each channel by a learnable scalar weight. Helps the model 
    dynamically prioritize certain feature channels during the bypass operation.
    """
    in_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # weight shape: (1, ..., 1, in_channels) for broadcasting
        # Standard Flax Dense logic handles channels last.
        # We want a learnable weight per channel.
        shape = [1] * x.ndim
        shape[-1] = self.in_channels
        
        w = self.param('weight', nn.initializers.ones, shape)
        return x * w

class SkipConnection(nn.Module):
    """
    Generic Skip (Bypass) Connection.
    
    Provides a parallel path to the spectral transformations, capturing 
    local information and facilitating gradient flow. 
    Matches the flexibility of the standard neuraloperator implementation.
    """
    out_channels: int
    skip_type: Literal["identity", "linear", "soft-gating"] = "linear"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.skip_type == "identity":
            return x
        elif self.skip_type == "linear":
            return nn.Dense(self.out_channels, use_bias=False)(x)
        elif self.skip_type == "soft-gating":
            return SoftGating(in_channels=self.out_channels)(x)
        else:
            raise ValueError(f"Unknown skip_type: {self.skip_type}")
