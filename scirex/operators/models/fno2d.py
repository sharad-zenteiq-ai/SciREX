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

from typing import Tuple, Callable
from flax import linen as nn
import jax.numpy as jnp
from ..layers.lifting import Lifting
from ..layers.projection import Projection
from ..blocks.fno_block import SpectralBlock

class FNO2D(nn.Module):
    """
    2D Fourier Neural Operator model.
    Structure: Lifting -> n_layers x SpectralBlock -> Projection
    """
    hidden_channels: int
    n_layers: int
    n_modes: Tuple[int, int]
    out_channels: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, nx, ny, in_channels)
        returns: (batch, nx, ny, out_channels)
        """
        # Lifting: encoder Dense -> project to hidden_channels
        x = Lifting(hidden_channels=self.hidden_channels)(x)
        
        # Iterative FNO blocks
        for _ in range(self.n_layers):
            x = SpectralBlock(
                hidden_channels=self.hidden_channels, 
                n_modes=self.n_modes,
                activation=self.activation
            )(x)
            
        # Projection: decoder Dense -> project to desired output channels
        x = Projection(out_channels=self.out_channels)(x)
        return x
