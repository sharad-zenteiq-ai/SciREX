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

from typing import Optional, Callable, Sequence
from flax import linen as nn
import jax.numpy as jnp

class ChannelMLP(nn.Module):
    """
    Unified Channel MLP (Point-wise MLP) for GINO and FNO.
    Standardized to JAX/Flax channels-last convention (B, ..., C).

    This implementation combines the efficiency of the FNO version with 
    additional features (dropout) often required for GINO.

    Attributes:
        out_channels (int): Dimensionality of the output representation.
        hidden_channels (int, optional): Width of internal layers. Defaults to out_channels.
        n_layers (int): Total number of linear transformations.
        activation (Callable): Activation function between layers.
        dropout_rate (float): Dropout probability.
    """
    out_channels: int
    hidden_channels: Optional[int] = None
    n_layers: int = 1
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden = self.hidden_channels if self.hidden_channels is not None else self.out_channels
        
        for i in range(self.n_layers):
            is_last = (i == self.n_layers - 1)
            layer_out = self.out_channels if is_last else hidden
            
            x = nn.Dense(layer_out, name=f"dense_{i}")(x)
            
            if not is_last:
                x = self.activation(x)
                if self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate, name=f"dropout_{i}")(
                        x, deterministic=deterministic
                    )
                    
        return x

class LinearChannelMLP(nn.Module):
    """
    Linear Channel MLP with explicit layer sizes.
    
    Attributes:
        layers (Sequence[int]): List of channel sizes. The first element is treated 
            as input (ignored by nn.Dense) and subsequent as layer widths.
        activation (Callable): Activation function.
        dropout_rate (float): Dropout probability.
    """
    layers: Sequence[int]
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        n_layers = len(self.layers) - 1
        assert n_layers >= 1, "Layers sequence must contain at least [in_channels, out_channels]"

        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            x = nn.Dense(self.layers[i + 1], name=f"dense_{i}")(x)

            if not is_last:
                x = self.activation(x)
                if self.dropout_rate > 0.0:
                    x = nn.Dropout(rate=self.dropout_rate, name=f"dropout_{i}")(
                        x, deterministic=deterministic
                    )

        return x