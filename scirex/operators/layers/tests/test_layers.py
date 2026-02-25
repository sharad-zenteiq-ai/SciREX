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
from flax import linen as nn
from scirex.operators.layers.lifting import Lifting
from scirex.operators.layers.projection import Projection

def test_lifting_shape():
    """Test Lifting layer forward pass shape."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 8, 8, 3
    hidden_channels = 16
    
    model = Lifting(hidden_channels=hidden_channels)
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, hidden_channels)

def test_projection_single_layer():
    """Test Projection layer with single-layer configuration."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 8, 8, 16
    out_channels = 5
    
    model = Projection(out_channels=out_channels)
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)

@pytest.mark.parametrize("activation", [nn.gelu, nn.relu, jax.nn.silu])
def test_projection_multi_layer(activation):
    """Test Projection layer with two-layer MLP configuration."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 2, 8, 8, 16
    hidden_dim = 32
    out_channels = 5
    
    model = Projection(out_channels=out_channels, hidden_dim=hidden_dim, activation=activation)
    x = jnp.ones((batch, nx, ny, in_channels))
    
    params = model.init(rng, x)
    y = model.apply(params, x)
    
    assert y.shape == (batch, nx, ny, out_channels)
