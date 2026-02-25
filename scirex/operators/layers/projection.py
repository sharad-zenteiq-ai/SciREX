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

class Projection(nn.Module):
    """
    Projection layer: projects the hidden representation to the desired output channels.
    
    Supports two modes:
    - Single-layer: Dense(hidden_channels -> out_channels)
    - Two-layer MLP: Dense(hidden_channels -> hidden_dim) + activation + Dense(hidden_dim -> out_channels)
      This matches the original WNO paper's projection (fc1 + GeLU + fc2).
    
    Parameters
    ----------
    out_channels : int
        Number of output channels.
    hidden_dim : int, optional
        If provided, uses a 2-layer MLP projection with this hidden dimension.
        The original WNO paper uses hidden_dim=192.
    activation : Callable
        Activation function between the two layers. Default is nn.gelu.
    """
    out_channels: int
    hidden_dim: Optional[int] = None
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.hidden_dim is not None:
            x = nn.Dense(self.hidden_dim)(x)
            x = self.activation(x)
        return nn.Dense(self.out_channels)(x)
