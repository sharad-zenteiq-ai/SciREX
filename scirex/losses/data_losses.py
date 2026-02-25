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
Standard data-driven loss functions for operator learning.
"""
import jax.numpy as jnp

def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error."""
    return jnp.mean((pred - target) ** 2)

def lp_loss(pred: jnp.ndarray, target: jnp.ndarray, p: int = 2) -> jnp.ndarray:
    """
    Relative Lp loss: ||pred - target||_p / ||target||_p
    Standard objective for operator learning.
    """
    # Flatten spatial and channel dimensions to (batch, -1)
    batch = pred.shape[0]
    diff = (pred - target).reshape(batch, -1)
    targ = target.reshape(batch, -1)
    
    # Compute norms per sample in batch
    diff_norm = jnp.linalg.norm(diff, ord=p, axis=1)
    targ_norm = jnp.linalg.norm(targ, ord=p, axis=1)
    
    # Return mean relative error over batch
    return jnp.mean(diff_norm / (targ_norm + 1e-8))
