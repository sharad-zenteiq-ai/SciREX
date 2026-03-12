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

"""
Unit tests for ChannelMLP and LinearChannelMLP layers.
Tests are dimension-agnostic (N-D spatial inputs).
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.channel_mlp import ChannelMLP, LinearChannelMLP


@pytest.mark.parametrize(
    "spatial_shape",
    [
        (16, 16),          # 2D
        (8, 8, 8),         # 3D
        (4, 4, 4, 4),      # 4D
    ],
)
def test_channel_mlp_forward_nd(spatial_shape):
    """Test ChannelMLP forward pass for N-D spatial inputs."""
    
    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 4
    out_channels = 8

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = ChannelMLP(
        out_channels=out_channels,
        hidden_channels=16,
        n_layers=3,
        dropout_rate=0.0,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, out_channels)


@pytest.mark.parametrize(
    "spatial_shape",
    [
        (12, 12),
        (6, 6, 6),
    ],
)

def test_channel_mlp_default_hidden(spatial_shape):
    """Test ChannelMLP when hidden_channels is None."""
    
    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 5
    out_channels = 10

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = ChannelMLP(
        out_channels=out_channels,
        hidden_channels=None,
        n_layers=2,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, out_channels)


@pytest.mark.parametrize(
    "spatial_shape",
    [
        (10, 10),
        (6, 6, 6),
    ],
)

def test_channel_mlp_with_dropout(spatial_shape):
    """Test ChannelMLP with dropout enabled."""
    
    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 3
    out_channels = 6

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = ChannelMLP(
        out_channels=out_channels,
        hidden_channels=12,
        n_layers=2,
        dropout_rate=0.5,
    )

    params = model.init({"params": rng, "dropout": rng}, x)

    y = model.apply(
        params,
        x,
        deterministic=True,
        rngs={"dropout": rng},
    )

    assert y.shape == (batch, *spatial_shape, out_channels)


@pytest.mark.parametrize(
    "spatial_shape",
    [
        (16, 16),
        (8, 8, 8),
    ],
)

def test_linear_channel_mlp_forward_nd(spatial_shape):
    """Test LinearChannelMLP forward pass for N-D spatial inputs."""
    
    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 4
    layers = [in_channels, 16, 8]

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = LinearChannelMLP(layers=layers)

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, layers[-1])