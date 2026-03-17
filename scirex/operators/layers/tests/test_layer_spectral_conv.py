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

"""
Unit tests for SpectralConv.

Tests are written in N-D style so the layer works for
arbitrary spatial dimensionalities and validates integration
with custom einsum utilities.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.spectral_conv import SpectralConv


# Forward pass (N-D)

@pytest.mark.parametrize(
    "spatial_shape,n_modes",
    [
        ((16, 16), (6, 6)),       # 2D
        ((8, 8, 8), (4, 4, 4)),   # 3D
    ],
)
def test_spectral_conv_forward_nd(spatial_shape, n_modes):
    """SpectralConv should preserve spatial dimensions."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    in_channels = 4
    out_channels = 6

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = SpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, out_channels)


# Channel variations

@pytest.mark.parametrize(
    "in_channels,out_channels",
    [
        (3, 5),
        (6, 8),
    ],
)
def test_spectral_conv_channel_variations(in_channels, out_channels):
    """SpectralConv should handle different channel sizes."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    spatial_shape = (16, 16)

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = SpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=(6, 6),
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == (batch, *spatial_shape, out_channels)


# Custom initialization

def test_spectral_conv_custom_init_std():
    """SpectralConv should work with custom initialization scale."""

    rng = jax.random.PRNGKey(0)

    batch = 2
    spatial_shape = (12, 12)
    in_channels = 4
    out_channels = 4

    x = jnp.ones((batch, *spatial_shape, in_channels))

    model = SpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=(5, 5),
        init_std=0.1,
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert y.shape == x.shape


# Numerical stability (NEW - important)

def test_spectral_conv_random_input():
    """SpectralConv should produce stable outputs for random inputs."""

    key = jax.random.PRNGKey(0)

    batch = 2
    spatial_shape = (16, 16)
    in_channels = 3
    out_channels = 5

    key1, key2 = jax.random.split(key)

    x = jax.random.normal(key1, (batch, *spatial_shape, in_channels))

    model = SpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=(6, 6),
    )

    params = model.init(key2, x)
    y = model.apply(params, x)

    # Check shape
    assert y.shape == (batch, *spatial_shape, out_channels)

    # Check finite values (important)
    assert jnp.all(jnp.isfinite(y))


# Zero input (edge case)

def test_spectral_conv_zero_input():
    """Zero input should produce stable (finite) output."""

    rng = jax.random.PRNGKey(0)

    batch = 1
    spatial_shape = (16, 16)
    in_channels = 2
    out_channels = 2

    x = jnp.zeros((batch, *spatial_shape, in_channels))

    model = SpectralConv(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=(4, 4),
    )

    params = model.init(rng, x)
    y = model.apply(params, x)

    assert jnp.all(jnp.isfinite(y))