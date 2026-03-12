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
Unit tests for embedding layers.

Tests are written in an N-dimensional style so the modules
work for arbitrary spatial dimensions.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.embeddings import (
    GridEmbedding,
    SinusoidalEmbedding,
    RotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)



# GridEmbedding

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (16, 16),       # 2D
        (8, 8, 8),      # 3D
        (4, 4, 4, 4),   # 4D
    ],
)

def test_grid_embedding_nd(spatial_shape):
    """GridEmbedding should append spatial coordinates."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 3

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = GridEmbedding(
        grid_boundaries=((0.0, 1.0),) * len(spatial_shape)
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    expected_channels = in_channels + len(spatial_shape)

    assert y.shape == (batch, *spatial_shape, expected_channels)



# SinusoidalEmbedding

@pytest.mark.parametrize(
    "embedding_type",
    ["transformer", "nerf"],
)
@pytest.mark.parametrize(
    "spatial_shape",
    [
        (10, 10),
        (6, 6, 6),
    ],
)

def test_sinusoidal_embedding_nd(spatial_shape, embedding_type):
    """SinusoidalEmbedding should produce correct output channels."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 4
    num_freq = 3

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = SinusoidalEmbedding(
        num_frequencies=num_freq,
        embedding_type=embedding_type,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    expected_channels = in_channels * num_freq * 2

    assert y.shape == (batch, *spatial_shape, expected_channels)


# RotaryEmbedding

@pytest.mark.parametrize(
    "spatial_shape",
    [
        (16, 16),
        (8, 8, 8),
    ],
)

def test_rotary_embedding_nd(spatial_shape):
    """RotaryEmbedding should generate frequency tensors."""

    batch = 2
    spatial_dim = len(spatial_shape)

    coords = jnp.ones((batch, *spatial_shape, spatial_dim))

    model = RotaryEmbedding(dim=8)

    freqs = model.apply({}, coords)

    assert freqs.shape[:-1] == coords.shape[:-1]


# rotate_half

def test_rotate_half_shape():
    """rotate_half should preserve tensor shape."""

    x = jnp.ones((2, 8, 8, 16))

    y = rotate_half(x)

    assert y.shape == x.shape


# apply_rotary_pos_emb

def test_apply_rotary_pos_emb_shape():
    """apply_rotary_pos_emb should preserve tensor shape."""

    t = jnp.ones((2, 8, 8, 16))
    freqs = jnp.ones((2, 8, 8, 16))

    y = apply_rotary_pos_emb(t, freqs)

    assert y.shape == t.shape