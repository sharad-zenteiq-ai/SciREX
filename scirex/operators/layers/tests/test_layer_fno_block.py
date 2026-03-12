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
Unit tests for FNOBlock.

Tests are written in N-D style to ensure the block works
for arbitrary spatial dimensionalities.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.fno_block import FNOBlock



# Basic forward pass (N-D)

@pytest.mark.parametrize(
    "spatial_shape,n_modes",
    [
        ((16, 16), (6, 6)),       # 2D
        ((8, 8, 8), (4, 4, 4)),   # 3D
    ],
)

def test_fno_block_forward_nd(spatial_shape, n_modes):
    """FNOBlock should preserve spatial shape."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    channels = 16

    x = jnp.ones((batch, *spatial_shape, channels))

    model = FNOBlock(
        hidden_channels=channels,
        n_modes=n_modes,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape


# With normalization

@pytest.mark.parametrize(
    "spatial_shape,n_modes",
    [
        ((12, 12), (5, 5)),
        ((6, 6, 6), (3, 3, 3)),
    ],
)

def test_fno_block_with_norm(spatial_shape, n_modes):
    """FNOBlock should run when normalization is enabled."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    channels = 8

    x = jnp.ones((batch, *spatial_shape, channels))

    model = FNOBlock(
        hidden_channels=channels,
        n_modes=n_modes,
        use_norm=True,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape


# Without ChannelMLP refinement

@pytest.mark.parametrize(
    "spatial_shape,n_modes",
    [
        ((14, 14), (6, 6)),
        ((6, 6, 6), (3, 3, 3)),
    ],
)

def test_fno_block_without_channel_mlp(spatial_shape, n_modes):
    """FNOBlock should run when ChannelMLP is disabled."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    channels = 12

    x = jnp.ones((batch, *spatial_shape, channels))

    model = FNOBlock(
        hidden_channels=channels,
        n_modes=n_modes,
        use_channel_mlp=False,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape



# Different skip connection types

@pytest.mark.parametrize(
    "skip_type",
    ["identity", "linear", "soft-gating"],
)

def test_fno_block_skip_types(skip_type):
    """FNOBlock should support all skip connection types."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    spatial_shape = (16, 16)
    channels = 10

    x = jnp.ones((batch, *spatial_shape, channels))

    model = FNOBlock(
        hidden_channels=channels,
        n_modes=(6, 6),
        skip_type=skip_type,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape