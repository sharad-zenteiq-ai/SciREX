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

from typing import Tuple, Callable, Optional, Literal, Union, List
from flax import linen as nn
import jax.numpy as jnp
from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import GridEmbedding
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlock

class FNO(nn.Module):
    """
    N-dimensional Fourier Neural Operator (supports 2D and 3D).
    
    The spatial dimensionality is automatically inferred from the length of
    ``n_modes``:
      - ``len(n_modes) == 2`` → 2D FNO  (input shape: batch, nx, ny, channels)
      - ``len(n_modes) == 3`` → 3D FNO  (input shape: batch, nx, ny, nz, channels)
    
    FNO learns operators between functional spaces by parameterized 
    integral kernels in the Fourier domain. This implementation aligns 
    with the standard architecture used in the original neuraloperator
    repository.
    
    The architecture follows a lifting-operator-projection pipeline:
    [Input] -> Lifting (ChannelMLP) -> Iterative FNOBlocks -> Projection (ChannelMLP) -> [Output]

    Attributes:
        hidden_channels (int): Width of the latent spectral representation.
        n_layers (int): Number of stacked FNO blocks.
        n_modes (Tuple[int, ...]): Fourier modes to retain per spatial dimension.
            Length determines 2D vs 3D.
        out_channels (int): Dimensionality of the output field.
        lifting_channel_ratio (int): Expansion factor for the lifting layer.
        projection_channel_ratio (int): Expansion factor for the projection layer.
        use_grid (bool): Whether to append Cartesian coordinates to the input.
        fno_skip (str): Skip connection type for the spectral branch.
        channel_mlp_skip (str): Skip connection type for the MLP refinement.
        use_channel_mlp (bool): Whether to use the MLP refinement branch.
        padding (float/list): Amount of domain padding to handle non-periodic boundaries.
        use_norm (bool): Whether to use InstanceNorm for training stabilization.
        activation (Callable): Activation function for the entire network.
    """
    hidden_channels: int
    n_layers: int
    n_modes: Tuple[int, ...]
    out_channels: int
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    use_grid: bool = True
    fno_skip: Literal["identity", "linear", "soft-gating"] = "linear"
    channel_mlp_skip: Literal["identity", "linear", "soft-gating"] = "soft-gating"
    use_channel_mlp: bool = True
    padding: Union[float, List[float]] = 0.0
    use_norm: bool = False
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Executes the forward pass.
        
        Args:
            x (jnp.ndarray): Input tensor.
                - 2D: shape (batch, nx, ny, in_channels)
                - 3D: shape (batch, nx, ny, nz, in_channels)
            
        Returns:
            jnp.ndarray: Predicted field with the same spatial dimensions
            and ``out_channels`` as the last dimension.
        """
        n_dim = len(self.n_modes)
        original_shape = x.shape
        
        # Check if padding is needed (handles both float and list)
        needs_pad = (
            any(p > 0 for p in self.padding)
            if isinstance(self.padding, (list, tuple))
            else self.padding > 0
        )
        
        # 1. Positional Embedding (Grid)
        if self.use_grid:
            grid_boundaries = tuple((0.0, 1.0) for _ in range(n_dim))
            x = GridEmbedding(grid_boundaries=grid_boundaries)(x)
            
        # 2. Domain Padding (to handle non-periodic conditions)
        if needs_pad:
            pad_layer = DomainPadding(padding=self.padding)
            x = pad_layer(x)
            
        # Stage 3: Spectral lifting (Encoder)
        # Maps input channels (e.g., physical parameters + grid) to latent space
        lifting_hidden = self.hidden_channels * self.lifting_channel_ratio
        x = ChannelMLP(
            out_channels=self.hidden_channels,
            hidden_channels=lifting_hidden,
            n_layers=2,
            activation=self.activation,
            name = ""lifting_layer"
        )(x)
        
        # Stage 4: Iterative Kernel Integration (Processing)
        # Global information propagation through Fourier space
        for _ in range(self.n_layers):
            x = FNOBlock(
                hidden_channels=self.hidden_channels, 
                n_modes=self.n_modes,
                activation=self.activation,
                use_norm=self.use_norm,
                skip_type=self.fno_skip,
                channel_mlp_skip=self.channel_mlp_skip,
                use_channel_mlp=self.use_channel_mlp
            )(x)
            
        # Stage 5: Spectral projection (Decoder)
        # Maps latent representation back to the physical target space
        projection_hidden = self.hidden_channels * self.projection_channel_ratio
        x = ChannelMLP(
            out_channels=self.out_channels,
            hidden_channels=projection_hidden,
            n_layers=2,
            activation=self.activation,
            name = "projection_layer"
        )(x)
        
        # 6. Inverse Domain Padding (Crop)
        if needs_pad:
            x = pad_layer(x, inverse=True, original_shape=original_shape)
        
        return x

