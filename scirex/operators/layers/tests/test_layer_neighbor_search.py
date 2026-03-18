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
Unit tests for NeighborSearch.

Tests focus on:
- Output shapes
- Radius filtering
- Distance correctness
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.neighbor_search import NeighborSearch


# -----------------------------
# Basic shape test
# -----------------------------
@pytest.mark.parametrize(
    "num_data, num_queries, dim, K",
    [
        (10, 5, 3, 2),
        (20, 8, 4, 3),
    ],
)
def test_neighbor_search_shapes(num_data, num_queries, dim, K):
    """NeighborSearch should return correct CSR shapes."""

    data = jnp.ones((num_data, dim))
    queries = jnp.ones((num_queries, dim))

    model = NeighborSearch(max_neighbors=K)

    out = model(data, queries, radius=10.0)

    assert out["neighbors_index"].shape == (num_queries * K,)
    assert out["neighbors_mask"].shape == (num_queries * K,)
    assert out["neighbors_row_splits"].shape == (num_queries + 1,)


# -----------------------------
# Radius filtering test
# -----------------------------
def test_neighbor_search_radius_mask():
    """Mask should correctly filter neighbors based on radius."""

    data = jnp.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ])

    queries = jnp.array([
        [0.0, 0.0],
    ])

    model = NeighborSearch(max_neighbors=2)

    out = model(data, queries, radius=1.0)

    mask = out["neighbors_mask"]

    # Only first point should be within radius
    assert mask[0] == True
    assert mask[1] == False


# -----------------------------
# Distance output test
# -----------------------------
def test_neighbor_search_with_distance():
    """Should return distances when return_norm=True."""

    data = jnp.array([
        [0.0, 0.0],
        [3.0, 4.0],  # distance = 5
    ])

    queries = jnp.array([
        [0.0, 0.0],
    ])

    model = NeighborSearch(max_neighbors=2, return_norm=True)

    out = model(data, queries, radius=10.0)

    dists = out["neighbors_distance"]

    assert dists.shape == (2,)
    assert jnp.all(dists >= 0)


# -----------------------------
# Correct nearest neighbors test
# -----------------------------
def test_neighbor_search_correct_neighbors():
    """Nearest neighbors should be selected correctly."""

    data = jnp.array([
        [0.0],
        [1.0],
        [5.0],
    ])

    queries = jnp.array([
        [0.0],
    ])

    model = NeighborSearch(max_neighbors=2)

    out = model(data, queries, radius=10.0)

    indices = out["neighbors_index"]

    # Expected nearest: 0 and 1
    assert set(indices.tolist()) == {0, 1}


# -----------------------------
# JIT compatibility
# -----------------------------
def test_neighbor_search_jit():
    """NeighborSearch should be JIT compatible."""

    data = jnp.ones((10, 3))
    queries = jnp.ones((5, 3))

    model = NeighborSearch(max_neighbors=2)

    @jax.jit
    def run(data, queries):
        return model(data, queries, radius=5.0)

    out = run(data, queries)

    assert out["neighbors_index"].shape == (10,)