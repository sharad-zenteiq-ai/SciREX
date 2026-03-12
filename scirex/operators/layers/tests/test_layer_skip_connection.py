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
Unit tests for SkipConnection and SoftGating.

Tests are written in N-D style so the modules work for
arbitrary spatial dimensions.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.skip_connection import SoftGating, SkipConnection


# SoftGating

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (16, 16),        # 2D
        (8, 8, 8),       # 3D
    ],
)

def test_soft_gating_nd(spatial_shape):
    """SoftGating should preserve tensor shape."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    channels = 6

    x = jnp.ones((batch, *spatial_shape, channels))

    model = SoftGating(in_channels=channels)

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape


# SkipConnection identity

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (12, 12),
        (6, 6, 6),
    ],
)

def test_skip_connection_identity(spatial_shape):
    """Identity skip should return the input unchanged."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    channels = 5

    x = jnp.ones((batch, *spatial_shape, channels))

    model = SkipConnection(out_channels=channels, skip_type="identity")

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape
    assert jnp.allclose(y, x)


# SkipConnection linear

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (16, 16),
        (8, 8, 8),
    ],
)

def test_skip_connection_linear(spatial_shape):
    """Linear skip should project channels."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 4
    out_channels = 10

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = SkipConnection(out_channels=out_channels, skip_type="linear")

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, out_channels)


# SkipConnection soft-gating

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (14, 14),
        (6, 6, 6),
    ],
)

def test_skip_connection_soft_gating(spatial_shape):
    """Soft-gating skip should preserve shape."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    channels = 7

    x = jnp.ones((batch, *spatial_shape, channels))

    model = SkipConnection(out_channels=channels, skip_type="soft-gating")

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape


# Invalid skip type

def test_skip_connection_invalid_type():
    """Invalid skip type should raise ValueError."""

    x = jnp.ones((2, 16, 16, 4))

    model = SkipConnection(out_channels=4, skip_type="invalid")

    with pytest.raises(ValueError):
        params = model.init(jax.random.PRNGKey(0), x)
        model.apply(params, x)