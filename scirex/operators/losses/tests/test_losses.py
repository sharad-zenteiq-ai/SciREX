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
from scirex.operators.losses.data_losses import mse, lp_loss

def test_mse():
    """Test MSE loss calculation."""
    pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.array([[1.1, 1.9], [3.2, 3.8]])
    
    expected = jnp.mean((pred - target) ** 2)
    actual = mse(pred, target)
    
    assert jnp.allclose(actual, expected)

def test_lp_loss():
    """Test Relative Lp loss calculation."""
    # (batch, nx, channels)
    pred = jnp.ones((2, 4, 1)) * 2.0
    target = jnp.ones((2, 4, 1)) * 1.0
    
    # diff = 1.0, norm_2 = sqrt(1^2 * 4) = 2.0
    # target = 1.0, norm_2 = sqrt(1^2 * 4) = 2.0
    # relative = 2.0 / 2.0 = 1.0
    
    actual = lp_loss(pred, target, p=2)
    assert jnp.allclose(actual, 1.0)

def test_lp_loss_zero_target():
    """Test Lp loss with zero target to ensure numerical stability."""
    pred = jnp.ones((2, 4, 1))
    target = jnp.zeros((2, 4, 1))
    
    # Should not crash due to 1e-8 in denominator
    actual = lp_loss(pred, target, p=2)
    assert not jnp.isnan(actual)
    assert not jnp.isinf(actual)
