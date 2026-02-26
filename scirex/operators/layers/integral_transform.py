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

from typing import Callable, Optional, Union, List
from flax import linen as nn
import jax.numpy as jnp
from .channel_mlp import ChannelMLP

class IntegralTransform(nn.Module):
    """
    Integral Kernel Transform (GNO-style).
    Equivalent to neuraloperator's IntegralTransform.
    
    Computes \int_{D} k(x, y, f(y)) * f(y) dy or variants.
    
    NOTE: This layer is NOT used by the FNO models. It is provided as
    a skeleton / placeholder for future Graph Neural Operator (GNO)
    implementations in SciREX. A full GNO implementation requires
    graph-based scatter/gather operations; for grid-based operator
    learning, the FNO uses SpectralConv instead.
    """
    channels: int
    transform_type: str = "linear" # linear, nonlinear
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, 
                 f_y: Optional[jnp.ndarray] = None, weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Generic kernel integral transform.
        Logic:
        1. Parametrize kernel k via a ChannelMLP
        2. Aggregate information from neighbors
        """
        # Note: A full GNO implementation in JAX typically requires 
        # graph-based operations (scatter/gather).
        # For now, we provide the architectural skeleton matching neuraloperator.
        
        # In a standard FNO (grid), we use SpectralConv instead of this explicit form.
        # This layer acts as a placeholder/base for GNO implementations in SciREX.
        
        # Example of a pointwise kernel approximation if used on grids:
        kernel_branch = ChannelMLP(
            out_channels=self.channels,
            hidden_channels=self.channels * 2,
            n_layers=2,
            activation=self.activation
        )(x)
        
        if f_y is not None:
            return kernel_branch * f_y
            
        return kernel_branch
