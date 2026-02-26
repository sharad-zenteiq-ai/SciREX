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
from scirex.operators.layers.padding import DomainPadding

def test_domain_padding_2d():
    """Test 2D DomainPadding forward and inverse."""
    batch, nx, ny, channels = 2, 16, 16, 3
    padding = 0.25
    x = jnp.ones((batch, nx, ny, channels))
    
    model = DomainPadding(padding=padding)
    y = model.apply({}, x)
    
    # Check shape: 16 * 0.25 = 4. Symmetric padding adds 4 on each side, so 16 + 4 + 4 = 24.
    assert y.shape == (batch, 24, 24, channels)
    
    # Test inverse
    x_rec = model.apply({}, y, inverse=True, original_shape=x.shape)
    assert x_rec.shape == x.shape
    assert jnp.allclose(x, x_rec)

def test_domain_padding_3d():
    """Test 3D DomainPadding forward and inverse."""
    batch, nx, ny, nz, channels = 2, 8, 8, 8, 3
    padding = 0.125
    x = jnp.ones((batch, nx, ny, nz, channels))
    
    model = DomainPadding(padding=padding)
    y = model.apply({}, x)
    
    # 8 * 0.125 = 1. Symmetric padding: 8 + 1 + 1 = 10.
    assert y.shape == (batch, 10, 10, 10, channels)
    
    # Test inverse
    x_rec = model.apply({}, y, inverse=True, original_shape=x.shape)
    assert x_rec.shape == x.shape
    assert jnp.allclose(x, x_rec)

def test_domain_padding_list():
    """Test DomainPadding with per-dimension padding list."""
    batch, nx, ny, channels = 2, 16, 32, 3
    padding = [0.25, 0.125]
    x = jnp.ones((batch, nx, ny, channels))
    
    model = DomainPadding(padding=padding)
    y = model.apply({}, x)
    
    # nx: 16 + 16*0.25*2 = 16 + 8 = 24
    # ny: 32 + 32*0.125*2 = 32 + 8 = 40
    assert y.shape == (batch, 24, 40, channels)
