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

import jax.numpy as jnp
import pytest
from scirex.operators.training.normalizers import GaussianNormalizer, UnitGaussianNormalizer

def test_gaussian_normalizer():
    """Test GaussianNormalizer encoding and decoding."""
    # Create some data: (100, 16, 16, 1)
    # Mean should be roughly 0.5, Std roughly 0.28 for uniform [0, 1]
    data = jnp.array(jnp.linspace(0, 1, 1000).reshape(10, 10, 10, 1))
    
    normalizer = GaussianNormalizer(data)
    
    encoded = normalizer.encode(data)
    decoded = normalizer.decode(encoded)
    
    # Check if encoded mean is 0 and std is 1 (roughly)
    reduce_axes = tuple(range(len(encoded.shape) - 1))
    assert jnp.allclose(jnp.mean(encoded, axis=reduce_axes), 0.0, atol=1e-5)
    assert jnp.allclose(jnp.std(encoded, axis=reduce_axes), 1.0, atol=1e-5)
    
    # Check if decoding restores original data
    assert jnp.allclose(data, decoded, atol=1e-5)

def test_gaussian_normalizer_multichannel():
    """Test GaussianNormalizer with multiple channels."""
    batch, nx, ny, channels = 4, 8, 8, 2
    data = jnp.ones((batch, nx, ny, channels))
    data = data.at[..., 0].set(1.0)
    data = data.at[..., 1].set(2.0)
    
    normalizer = GaussianNormalizer(data)
    
    # Since all values are same in each channel, std is 0.
    # Normalizer adds eps = 1e-7.
    encoded = normalizer.encode(data)
    assert jnp.allclose(encoded, 0.0)
    
    decoded = normalizer.decode(encoded)
    assert jnp.allclose(data, decoded)

def test_unit_gaussian_normalizer():
    """Test UnitGaussianNormalizer alias."""
    data = jnp.array(jnp.linspace(0, 1, 100).reshape(4, 5, 5, 1))
    normalizer = UnitGaussianNormalizer(data)
    encoded = normalizer.encode(data)
    decoded = normalizer.decode(encoded)
    assert jnp.allclose(data, decoded, atol=1e-5)
