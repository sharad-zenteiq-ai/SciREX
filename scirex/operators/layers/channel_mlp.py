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

class ChannelMLP(nn.Module):
    """
    Point-wise Multi-layer Perceptron (Channel MLP).
    
    This module applies an MLP independently to each spatial location (pixel/voxel) 
    across the channel dimension. It is a core component of the modern FNO 
    refinement, helping the model learn complex, localized channel interactions.

    Attributes:
        out_channels (int): Dimensionality of the output representation.
        hidden_channels (int, optional): Width of the internal hidden layers.
        n_layers (int): Total number of linear transformations.
        activation (Callable): Activation function between layers.
    """
    out_channels: int
    hidden_channels: Optional[int] = None
    n_layers: int = 1
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden = self.hidden_channels if self.hidden_channels is not None else self.out_channels
        
        for i in range(self.n_layers):
            # Last layer projects to out_channels, others to hidden
            is_last = (i == self.n_layers - 1)
            layer_out = self.out_channels if is_last else hidden
            
            x = nn.Dense(layer_out)(x)
            
            if not is_last:
                x = self.activation(x)
                
        return x
