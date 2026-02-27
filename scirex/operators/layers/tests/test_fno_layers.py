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
Unit test for forward pass shape of the FNO2D model.
"""
import jax
import jax.numpy as jnp
from scirex.operators.models.fno import FNO2D
from scirex.training.train_state import create_train_state
from scirex.operators.layers.fno_block import FNOBlock, FNOBlock3D

def test_fno_block_shape():
    """Test 2D FNOBlock forward pass shape."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, channels = 2, 16, 16, 32
    n_modes = (8, 8)
    
    model = FNOBlock(hidden_channels=channels, n_modes=n_modes)
    x = jnp.ones((batch, nx, ny, channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, channels)

def test_fno_block_3d_shape():
    """Test 3D FNOBlock forward pass shape."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, nz, channels = 2, 8, 8, 8, 16
    n_modes = (4, 4, 4)
    
    model = FNOBlock3D(hidden_channels=channels, n_modes=n_modes)
    x = jnp.ones((batch, nx, ny, nz, channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, nz, channels)

def test_fno_block_configs():
    """Test FNOBlock with various flags (norm, no mlp)."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, channels = 2, 8, 8, 16
    n_modes = (4, 4)
    
    # Test with norm and without channel MLP
    model = FNOBlock(
        hidden_channels=channels, 
        n_modes=n_modes,
        use_norm=True,
        use_channel_mlp=False
    )
    x = jnp.ones((batch, nx, ny, channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, channels)

def test_forward_shape():
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 16, 16, 1
    hidden_channels = 16
    n_layers = 2
    n_modes = (6, 6)
    out_channels = 1

    model = FNO2D(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        n_modes=n_modes, 
        out_channels=out_channels
    )
    state = create_train_state(rng, model, (batch, nx, ny, in_channels), learning_rate=1e-3)
    x = jnp.ones((batch, nx, ny, in_channels), dtype=jnp.float32)
    
    preds = state.apply_fn({"params": state.params}, x)
    assert preds.shape == (batch, nx, ny, out_channels)
    print("test_forward_shape OK:", preds.shape)

if __name__ == "__main__":
    test_forward_shape()
