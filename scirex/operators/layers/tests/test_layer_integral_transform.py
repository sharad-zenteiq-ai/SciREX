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
Unit tests for IntegralTransform (CSR-based version).

Tests focus on:
- Forward pass correctness
- Shape consistency
"""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from scirex.operators.layers.integral_transform import IntegralTransform


# Helper: simple MLP

class SimpleMLP(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.out_channels)(x)



# Helper: build neighbors

def build_neighbors(num_nodes, K):
    idx = jnp.tile(jnp.arange(K), (num_nodes, 1)) % num_nodes
    splits = jnp.arange(0, (num_nodes * K) + 1, K)
    mask = jnp.ones((num_nodes, K))

    return {
        "neighbors_index": idx,
        "neighbors_row_splits": splits,
        "neighbors_mask": mask,
    }



# Forward pass

@pytest.mark.parametrize(
    "num_nodes, in_channels, out_channels, K",
    [
        (8, 4, 6, 2),
        (16, 3, 5, 3),
    ],
)
def test_integral_transform_forward(num_nodes, in_channels, out_channels, K):
    rng = jax.random.PRNGKey(0)

    y = jnp.ones((num_nodes, in_channels))
    neighbors = build_neighbors(num_nodes, K)

    model = IntegralTransform(
        channel_mlp=SimpleMLP(out_channels),
        in_channels=in_channels,
        out_channels=out_channels,
    )

    params = model.init(rng, y, neighbors)
    out = model.apply(params, y, neighbors)

    assert out.shape == (num_nodes, out_channels)



# Forward with f_y

@pytest.mark.parametrize(
    "num_nodes, in_channels, out_channels, K",
    [
        (10, 4, 6, 2),
        (12, 5, 7, 3),
    ],
)
def test_integral_transform_with_fy(num_nodes, in_channels, out_channels, K):
    rng = jax.random.PRNGKey(0)

    y = jnp.ones((num_nodes, in_channels))
    f_y = jnp.ones((num_nodes, in_channels))

    neighbors = build_neighbors(num_nodes, K)

    model = IntegralTransform(
        channel_mlp=SimpleMLP(out_channels),
        in_channels=in_channels,
        out_channels=out_channels,
    )

    params = model.init(rng, y, neighbors, f_y=f_y)
    out = model.apply(params, y, neighbors, f_y=f_y)

    assert out.shape == (num_nodes, out_channels)


# Mask handling

def test_integral_transform_with_mask():
    rng = jax.random.PRNGKey(0)

    num_nodes = 8
    in_channels = 4
    out_channels = 6
    K = 2

    y = jnp.ones((num_nodes, in_channels))
    neighbors = build_neighbors(num_nodes, K)

    neighbors["neighbors_mask"] = jnp.array([[1, 0]] * num_nodes)

    model = IntegralTransform(
        channel_mlp=SimpleMLP(out_channels),
        in_channels=in_channels,
        out_channels=out_channels,
        reduction="mean",
    )

    params = model.init(rng, y, neighbors)
    out = model.apply(params, y, neighbors)

    assert out.shape == (num_nodes, out_channels)



# Weight handling

def test_integral_transform_with_weights():
    rng = jax.random.PRNGKey(0)

    num_nodes = 8
    in_channels = 4
    out_channels = 6
    K = 2

    y = jnp.ones((num_nodes, in_channels))
    neighbors = build_neighbors(num_nodes, K)

    weights = jnp.ones((num_nodes * K,))

    model = IntegralTransform(
        channel_mlp=SimpleMLP(out_channels),
        in_channels=in_channels,
        out_channels=out_channels,
    )

    params = model.init(rng, y, neighbors, weights=weights)
    out = model.apply(params, y, neighbors, weights=weights)

    assert out.shape == (num_nodes, out_channels)