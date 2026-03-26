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

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scirex.operators.models.fnogno import FNOGNO

def test_fnogno_forward():
    """Test FNOGNO forward pass shapes."""
    batch, nx, ny, nz, in_channels = 1, 8, 8, 8, 3
    n_out, out_channels = 10, 2
    coord_dim = 3
    
    # ── 1. Create Dummy Inputs ──
    x_grid = jnp.linspace(0, 1, nx)
    y_grid = jnp.linspace(0, 1, ny)
    z_grid = jnp.linspace(0, 1, nz)
    grid = jnp.stack(jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij'), axis=-1)
    # (batch, nx, ny, nz, 3)
    in_p = jnp.repeat(grid[None, ...], batch, axis=0)
    
    # (batch, nx, ny, nz, in_channels)
    f = jax.random.normal(jax.random.PRNGKey(0), (batch, nx, ny, nz, in_channels))
    
    # (batch, n_out, 3)
    out_p = jax.random.uniform(jax.random.PRNGKey(1), (batch, n_out, coord_dim))
    
    # ── 2. Initialize Model ──
    model = FNOGNO(
        in_channels=in_channels,
        out_channels=out_channels,
        fno_n_modes=(4, 4, 4),
        fno_hidden_channels=16,
        fno_n_layers=2,
        gno_coord_dim=coord_dim,
        gno_radius=0.5
    )
    
    rng = jax.random.PRNGKey(42)
    variables = model.init(rng, in_p=in_p, out_p=out_p, f=f)
    
    # ── 3. Forward Pass ──
    y_out = model.apply(variables, in_p=in_p, out_p=out_p, f=f)
    
    # ── 4. Verify ──
    assert y_out.shape == (batch, n_out, out_channels)
    assert not jnp.isnan(y_out).any()

if __name__ == "__main__":
    pytest.main([__file__])
