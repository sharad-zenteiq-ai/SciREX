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

from typing import Optional, Callable, Tuple, Literal
from flax import linen as nn
import jax.numpy as jnp
from .spectral_conv import SpectralConv
from .skip_connection import SkipConnection
from .channel_mlp import ChannelMLP

class FNOBlock(nn.Module):
    """
    N-dimensional Fourier Neural Operator (FNO) Block (supports 2D and 3D).
    
    The dimensionality is automatically inferred from the length of ``n_modes``:
      - ``len(n_modes) == 2`` → 2D FNO block
      - ``len(n_modes) == 3`` → 3D FNO block
    
    This block implements a single iterative step of the FNO architecture, 
    combining a spectral convolution for global feature learning with a 
    standard convolution (via SkipConnection) for local feature refinement.
    
    Attributes:
        hidden_channels (int): Dimensionality of the latent representation.
        n_modes (Tuple[int, ...]): Number of Fourier modes to retain in each
            spatial dimension. Length determines 2D vs 3D.
        activation (Callable): Nonlinear activation function (default: nn.gelu).
        use_norm (bool): Whether to apply Instance Normalization for training stability.
        skip_type (str): Type of bypass connection ('identity', 'linear', or 'soft-gating').
        channel_mlp_skip (str): Type of bypass connection for the Channel MLP refinement.
        use_channel_mlp (bool): Whether to include a point-wise MLP after the spectral layers.
    """
    hidden_channels: int
    n_modes: Tuple[int, ...]
    activation: Callable = nn.gelu
    use_norm: bool = False
    skip_type: Literal["identity", "linear", "soft-gating"] = "linear"
    channel_mlp_skip: Literal["identity", "linear", "soft-gating"] = "soft-gating"
    use_channel_mlp: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Step 1: Global feature extraction via Spectral Convolution
        y_s = SpectralConv(
            in_channels=self.hidden_channels, 
            out_channels=self.hidden_channels, 
            n_modes=self.n_modes
        )(x)
        
        # Step 2: Local feature extraction via Spatial Skip Connection
        y_p = SkipConnection(
            out_channels=self.hidden_channels, 
            skip_type=self.skip_type
        )(x)
        
        # Step 3: Branch fusion and stabilization
        x = y_s + y_p
        
        if self.use_norm:
            x = nn.InstanceNorm()(x)
        
        x = self.activation(x)

        # Step 4: Point-wise refinement (Channel MLP)
        # Often referred to as the 'modern' FNO variant or 'FNO-MLP'
        if self.use_channel_mlp:
            y_mlp = ChannelMLP(
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=2,
                activation=self.activation
            )(x)
            
            y_skip = SkipConnection(
                out_channels=self.hidden_channels,
                skip_type=self.channel_mlp_skip
            )(x)
            
            x = y_mlp + y_skip
            x = self.activation(x)

        return x

