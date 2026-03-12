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
Unit tests for the FNO model architecture.
Verifies forward pass shapes and parameter consistency for both 2D and 3D
using the single unified FNO class with different n_modes.
"""

import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import pytest
from scirex.operators.models.fno import FNO

@pytest.mark.parametrize("n_modes", [(8, 8), (4, 4)])
@pytest.mark.parametrize("hidden_channels", [32, 64])
def test_fno_2d_forward(n_modes, hidden_channels):
    """Test FNO forward pass shape with 2D n_modes."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 16, 16, 1
    n_layers = 4
    out_channels = 1
    
    model = FNO(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        n_modes=n_modes, 
        out_channels=out_channels
    )
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)

@pytest.mark.parametrize("padding", [0.0, 0.1])
def test_fno_3d_forward(padding):
    """Test FNO forward pass shape with 3D n_modes and optional padding."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, nz, in_channels = 2, 8, 8, 8, 1
    hidden_channels, n_layers = 16, 2
    n_modes = (4, 4, 4)
    out_channels = 1
    
    model = FNO(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        n_modes=n_modes, 
        out_channels=out_channels,
        padding=padding
    )
    x = jnp.ones((batch, nx, ny, nz, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, nz, out_channels)

def test_fno_variable_channels():
    """Test FNO with different input and output channels (2D)."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 16, 16, 3
    hidden_channels, n_layers = 32, 2
    n_modes = (4, 4)
    out_channels = 5
    
    model = FNO(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        n_modes=n_modes, 
        out_channels=out_channels
    )
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-c", "/dev/null"]))