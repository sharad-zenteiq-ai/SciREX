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
Unit tests for IntegralTransform.

Tests are written in N-D style so the layer works for
arbitrary spatial dimensionalities.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.integral_transform import IntegralTransform


# Forward pass without f_y

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (16, 16),        # 2D
        (8, 8, 8),       # 3D
    ],
)

def test_integral_transform_forward_nd(spatial_shape):
    """IntegralTransform should run and preserve spatial shape."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 4
    channels = 6

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = IntegralTransform(channels=channels)

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, channels)


# Forward pass with f_y

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (12, 12),
        (6, 6, 6),
    ],
)

def test_integral_transform_with_fy(spatial_shape):
    """IntegralTransform should handle f_y branch."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 5
    channels = 7

    x = jnp.ones((batch, *spatial_shape, in_channels))
    f_y = jnp.ones((batch, *spatial_shape, channels))

    model = IntegralTransform(channels=channels)

    params = model.init(rng, x)

    y = model.apply(params, x, f_y=f_y)

    assert y.shape == (batch, *spatial_shape, channels)


# Different channel configurations

@pytest.mark.parametrize(
    "channels",
    [4, 8, 12],
)

def test_integral_transform_channel_variations(channels):
    """IntegralTransform should support different channel sizes."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    spatial_shape = (16, 16)
    in_channels = 3

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = IntegralTransform(channels=channels)

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, channels)