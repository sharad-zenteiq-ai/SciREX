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
Unit tests for the physical Relative L2 loss computation.
"""

import jax.numpy as jnp
import pytest
from scirex.operators.losses.rel_l2_losses import phys_rel_l2_loss

class MockNormalizer:
    """Mock normalizer for testing phys_rel_l2_loss."""
    def decode(self, x):
        return x * 2.0

def test_phys_rel_l2_loss():
    """Test the physical Relative L2 loss computation."""
    normalizer = MockNormalizer()
    
    # Create simple 2D arrays (batch=2, features=2)
    pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    
    # 1. Zero loss case
    loss_zero = phys_rel_l2_loss(pred, target, normalizer)
    assert jnp.allclose(loss_zero, 0.0)
    
    # 2. Non-zero loss case
    pred_diff = jnp.array([[1.1, 2.1], [3.1, 4.1]])
    loss_val = phys_rel_l2_loss(pred_diff, target, normalizer)
    
    # Verification:
    # pred_decoded = [2.2, 4.2], [6.2, 8.2]
    # target_decoded = [2.0, 4.0], [6.0, 8.0]
    # diff = [0.2, 0.2], [0.2, 0.2]
    # batch 0: diff_norm = sqrt(0.2^2 + 0.2^2) = sqrt(0.08)
    # batch 0: target_norm = sqrt(2.0^2 + 4.0^2) = sqrt(20)
    # rel_l2_0 = sqrt(0.08 / 20) = sqrt(0.004)
    
    # batch 1: diff_norm = sqrt(0.08)
    # batch 1: target_norm = sqrt(6^2 + 8^2) = 10
    # rel_l2_1 = sqrt(0.08) / 10
    
    expected_rel_0 = jnp.sqrt(0.08) / jnp.sqrt(20.0)
    expected_rel_1 = jnp.sqrt(0.08) / 10.0
    expected_avg = (expected_rel_0 + expected_rel_1) / 2.0
    
    assert jnp.allclose(loss_val, expected_avg, atol=1e-6)

if __name__ == "__main__":
    test_phys_rel_l2_loss()
