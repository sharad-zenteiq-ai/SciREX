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

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from scirex.operators.layers.channel_mlp import ChannelMLP
from scirex.operators.layers.skip_connection import SkipConnection, SoftGating
from scirex.operators.layers.integral_transform import IntegralTransform
from scirex.operators.layers.embeddings import GridEmbedding

def test_grid_embedding_2d():
    """Test GridEmbedding with 2D input."""
    batch, nx, ny, channels = 2, 8, 8, 3
    x = jnp.ones((batch, nx, ny, channels))
    
    model = GridEmbedding(grid_boundaries=((0.0, 1.0), (0.0, 1.0)))
    y = model.apply({}, x)
    
    # original channels (3) + 2 grid channels = 5
    assert y.shape == (batch, nx, ny, channels + 2)
    
    # Check if grid values are correct (corners)
    # x-coords are in y[..., 3], y-coords in y[..., 4]
    assert jnp.allclose(y[0, 0, 0, 3:], jnp.array([0.0, 0.0]))
    assert jnp.allclose(y[0, -1, -1, 3:], jnp.array([1.0, 1.0]))

def test_grid_embedding_3d():
    """Test GridEmbedding with 3D input."""
    batch, nx, ny, nz, channels = 2, 4, 4, 4, 1
    x = jnp.ones((batch, nx, ny, nz, channels))
    
    model = GridEmbedding(grid_boundaries=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    y = model.apply({}, x)
    
    # original channels (1) + 3 grid channels = 4
    assert y.shape == (batch, nx, ny, nz, channels + 3)
    
    # Check corners
    assert jnp.allclose(y[0, 0, 0, 0, 1:], jnp.array([0.0, 0.0, 0.0]))
    assert jnp.allclose(y[0, -1, -1, -1, 1:], jnp.array([1.0, 1.0, 1.0]))

def test_channel_mlp_lifting_shape():
    """Test ChannelMLP in a lifting configuration (single layer)."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 8, 8, 3
    out_channels = 16
    
    model = ChannelMLP(out_channels=out_channels, n_layers=1)
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)

def test_channel_mlp_projection_shape():
    """Test ChannelMLP in a projection configuration (two layers)."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 8, 8, 16
    hidden_channels = 32
    out_channels = 5
    
    model = ChannelMLP(
        out_channels=out_channels, 
        hidden_channels=hidden_channels, 
        n_layers=2
    )
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)

@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("activation", [nn.gelu, nn.relu, jax.nn.silu])
def test_channel_mlp_flexible_configs(n_layers, activation):
    """Test ChannelMLP with various layering and activations."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 4, 4, 8
    hidden_channels = 16
    out_channels = 4
    
    model = ChannelMLP(
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        activation=activation
    )
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)

@pytest.mark.parametrize("skip_type", ["identity", "linear", "soft-gating"])
def test_skip_connection_shapes(skip_type):
    """Test SkipConnection for different types."""
    rng = jax.random.PRNGKey(0)
    batch, res, channels = 2, 16, 32
    x = jnp.ones((batch, res, res, channels))
    
    # For soft-gating and identity, in/out MUST match.
    # For linear, they can differ, but we keep them same for consistency.
    model = SkipConnection(out_channels=channels, skip_type=skip_type)
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, res, res, channels)

def test_soft_gating_behavior():
    """Verify SoftGating learnable parameter existence and shape."""
    rng = jax.random.PRNGKey(0)
    batch, channels = 2, 16
    x = jnp.ones((batch, 8, 8, channels))
    
    model = SoftGating(in_channels=channels)
    params = model.init(rng, x)
    
    assert 'weight' in params['params']
    assert params['params']['weight'].shape == (1, 1, 1, channels)

def test_integral_transform_skeleton():
    """Verify IntegralTransform skeleton can be initialized and called."""
    rng = jax.random.PRNGKey(0)
    batch, res, channels = 2, 16, 32
    x = jnp.ones((batch, res, res, channels))
    
    model = IntegralTransform(channels=channels)
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, res, res, channels)
