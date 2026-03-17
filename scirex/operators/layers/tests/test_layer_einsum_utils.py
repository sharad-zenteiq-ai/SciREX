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

from scirex.operators.layers.einsum_utils import (
    einsum,
    _einsum_complexhalf_two_input,
    _einsum_complexhalf_general,
)


def test_matrix_multiplication_complex64():
    """Test standard complex64 einsum (should use native JAX)."""
    a = jnp.array([[1+2j, 3+4j]], dtype=jnp.complex64)
    b = jnp.array([[5+6j], [7+8j]], dtype=jnp.complex64)

    expected = jnp.einsum("ij,jk->ik", a, b)
    result = einsum("ij,jk->ik", a, b)

    assert jnp.allclose(expected, result, atol=1e-5)


def test_complex_half_precision_two_input():
    """Test custom half-precision path explicitly (2 inputs)."""
    a = jnp.array([[1+2j, 3+4j]], dtype=jnp.complex64)
    b = jnp.array([[5+6j], [7+8j]], dtype=jnp.complex64)

    expected = jnp.einsum("ij,jk->ik", a, b)
    result = _einsum_complexhalf_two_input("ij,jk->ik", a, b)

    # Allow tolerance due to float16 precision
    assert jnp.allclose(result, expected, atol=1e-2)


def test_dot_product():
    """Test vector dot product."""
    a = jnp.array([1+2j, 3+4j], dtype=jnp.complex64)
    b = jnp.array([5+6j, 7+8j], dtype=jnp.complex64)

    result = einsum("i,i->", a, b)
    expected = jnp.einsum("i,i->", a, b)

    assert jnp.allclose(result, expected, atol=1e-5)


def test_multiple_inputs():
    """Test multi-input einsum using custom implementation."""
    a = jnp.ones((2, 2), dtype=jnp.complex64)
    b = jnp.ones((2, 2), dtype=jnp.complex64)
    c = jnp.ones((2, 2), dtype=jnp.complex64)

    result = einsum("ij,jk,kl->il", a, b, c)
    expected = jnp.einsum("ij,jk,kl->il", a, b, c)

    assert jnp.allclose(result, expected, atol=1e-5)


def test_complex_half_precision_general():
    """Test custom half-precision path for multiple inputs."""
    a = jnp.ones((2, 2), dtype=jnp.complex64)
    b = jnp.ones((2, 2), dtype=jnp.complex64)
    c = jnp.ones((2, 2), dtype=jnp.complex64)

    expected = jnp.einsum("ij,jk,kl->il", a, b, c)
    result = _einsum_complexhalf_general("ij,jk,kl->il", a, b, c)

    assert jnp.allclose(result, expected, atol=1e-2)


def test_real_inputs():
    """Test that real tensors use native einsum."""
    a = jnp.array([[1., 2.], [3., 4.]])
    b = jnp.array([[5., 6.], [7., 8.]])

    result = einsum("ij,jk->ik", a, b)
    expected = jnp.einsum("ij,jk->ik", a, b)

    assert jnp.allclose(result, expected)


def test_zero_tensor():
    """Edge case: zero tensors."""
    a = jnp.zeros((2, 2), dtype=jnp.complex64)
    b = jnp.zeros((2, 2), dtype=jnp.complex64)

    result = einsum("ij,jk->ik", a, b)
    expected = jnp.zeros((2, 2), dtype=jnp.complex64)

    assert jnp.allclose(result, expected)


def test_jit_compilation():
    """Test JIT compatibility."""
    a = jnp.array([[1+2j, 3+4j]], dtype=jnp.complex64)
    b = jnp.array([[5+6j], [7+8j]], dtype=jnp.complex64)

    jit_fn = jax.jit(einsum, static_argnames=["eq"])

    result = jit_fn("ij,jk->ik", a, b)
    expected = jnp.einsum("ij,jk->ik", a, b)

    assert jnp.allclose(result, expected, atol=1e-5)


def test_random_inputs_half_precision():
    """Random test for numerical stability."""
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    a = jax.random.normal(key1, (4, 4)) + 1j * jax.random.normal(key1, (4, 4))
    b = jax.random.normal(key2, (4, 4)) + 1j * jax.random.normal(key2, (4, 4))

    a = a.astype(jnp.complex64)
    b = b.astype(jnp.complex64)

    expected = jnp.einsum("ij,jk->ik", a, b)
    result = _einsum_complexhalf_two_input("ij,jk->ik", a, b)

    assert jnp.allclose(result, expected, atol=1e-2)