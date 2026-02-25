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
from scirex.operators.blocks.fno_block import SpectralBlock, SpectralBlock3D

@pytest.mark.parametrize("hidden_channels", [16, 32])
def test_spectral_block_shape(hidden_channels):
    """Test SpectralBlock (2D) forward pass shape."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny = 2, 16, 16
    n_modes = (4, 4)
    
    model = SpectralBlock(hidden_channels=hidden_channels, n_modes=n_modes)
    x = jnp.ones((batch, nx, ny, hidden_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, hidden_channels)

def test_spectral_block_with_norm():
    """Test SpectralBlock (2D) with normalization enabled."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, hidden_channels = 2, 16, 16, 16
    n_modes = (4, 4)
    
    model = SpectralBlock(hidden_channels=hidden_channels, n_modes=n_modes, use_norm=True)
    x = jnp.ones((batch, nx, ny, hidden_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, hidden_channels)

def test_spectral_block3d_shape():
    """Test SpectralBlock3D forward pass shape."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, nz, hidden_channels = 2, 8, 8, 8, 16
    n_modes = (4, 4, 4)
    
    model = SpectralBlock3D(hidden_channels=hidden_channels, n_modes=n_modes)
    x = jnp.ones((batch, nx, ny, nz, hidden_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, nz, hidden_channels)
