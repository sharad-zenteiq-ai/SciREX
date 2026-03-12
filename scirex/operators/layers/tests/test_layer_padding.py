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
Unit tests for DomainPadding.

Tests are written in N-D style to ensure padding works
for arbitrary spatial dimensions.
"""

import jax.numpy as jnp
import pytest

from scirex.operators.layers.padding import DomainPadding


# Forward padding (N-D)

@pytest.mark.parametrize(
    "spatial_shape,padding",
    [
        ((16, 16), 0.25),        # 2D
        ((8, 8, 8), 0.125),      # 3D
    ],
)

def test_domain_padding_forward_nd(spatial_shape, padding):
    """Padding should expand spatial dimensions symmetrically."""

    batch = 2
    channels = 3

    x = jnp.ones((batch, *spatial_shape, channels))

    model = DomainPadding(padding=padding)

    y = model.apply({}, x)

    # compute expected padded shape
    padded_dims = []
    for d in spatial_shape:
        p = int(round(d * padding))
        padded_dims.append(d + 2 * p)

    assert y.shape == (batch, *padded_dims, channels)


# Inverse padding (crop)

@pytest.mark.parametrize(
    "spatial_shape,padding",
    [
        ((16, 16), 0.25),
        ((8, 8, 8), 0.125),
    ],
)

def test_domain_padding_inverse_nd(spatial_shape, padding):
    """Inverse padding should restore the original tensor."""

    batch = 2
    channels = 3

    x = jnp.ones((batch, *spatial_shape, channels))

    model = DomainPadding(padding=padding)

    padded = model.apply({}, x)

    restored = model.apply({}, padded, inverse=True, original_shape=x.shape)

    assert restored.shape == x.shape
    assert jnp.allclose(restored, x)


# List padding per dimension

def test_domain_padding_list_padding():
    """Padding specified per dimension should work."""

    batch = 2
    spatial_shape = (16, 32)
    channels = 3

    padding = [0.25, 0.125]

    x = jnp.ones((batch, *spatial_shape, channels))

    model = DomainPadding(padding=padding)

    y = model.apply({}, x)

    expected_x = 16 + 2 * int(round(16 * 0.25))
    expected_y = 32 + 2 * int(round(32 * 0.125))

    assert y.shape == (batch, expected_x, expected_y, channels)


# Error handling

def test_domain_padding_inverse_requires_shape():
    """Inverse padding must raise error if original_shape is missing."""

    x = jnp.ones((2, 16, 16, 3))

    model = DomainPadding(padding=0.25)

    padded = model.apply({}, x)

    with pytest.raises(ValueError):
        model.apply({}, padded, inverse=True)