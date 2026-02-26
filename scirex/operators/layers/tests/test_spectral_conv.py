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
from scirex.operators.layers.spectral_conv import SpectralConv2D, SpectralConv3D

@pytest.mark.parametrize("n_modes", [(4, 4), (8, 4)])
@pytest.mark.parametrize("out_channels", [5, 10])
def test_spectral_conv2d_shape(n_modes, out_channels):
    """Test SpectralConv2D forward pass shape."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 16, 16, 3
    
    model = SpectralConv2D(in_channels=in_channels, out_channels=out_channels, n_modes=n_modes)
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)

def test_spectral_conv3d_shape():
    """Test SpectralConv3D forward pass shape."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, nz, in_channels = 2, 8, 8, 8, 3
    out_channels = 5
    n_modes = (4, 4, 4)
    
    model = SpectralConv3D(in_channels=in_channels, out_channels=out_channels, n_modes=n_modes)
    x = jnp.ones((batch, nx, ny, nz, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, nz, out_channels)

@pytest.mark.parametrize("nx, ny", [(16, 16), (32, 16), (16, 32)])
def test_spectral_conv2d_various_resolutions(nx, ny):
    """Test SpectralConv2D with different input resolutions."""
    rng = jax.random.PRNGKey(0)
    batch, in_channels, out_channels = 2, 4, 4
    n_modes = (4, 4)
    
    model = SpectralConv2D(in_channels=in_channels, out_channels=out_channels, n_modes=n_modes)
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)
