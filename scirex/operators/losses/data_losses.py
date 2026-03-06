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


def h1_loss(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Relative H1 Sobolev loss.

    Matches the ``neuraloperator`` ``H1Loss(d=2)`` convention:

        per_sample = sqrt(diff) / (sqrt(ynorm) + eps)

    where ``diff`` and ``ynorm`` are the squared L2 norms of the
    value *and* first-order spatial gradients (central finite
    differences with periodic BCs).  Reduction: ``sum`` over the
    batch (reference default).

    The function is agnostic to the number of spatial dimensions
    beyond the first two (e.g. an extra trivial z-dim is fine).

    Args:
        pred: Predictions, shape ``(batch, nx, ny, ...)``.
        target: Ground truth, same shape as *pred*.
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar H1 loss.
    """
    # L2 part ----------------------------------------------------------
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    diff_l2 = jnp.sum((pred_flat - target_flat) ** 2, axis=-1)
    ynorm_l2 = jnp.sum(target_flat ** 2, axis=-1)

    # Spatial derivatives via periodic central differences (x=dim1, y=dim2)
    dx_pred = jnp.roll(pred, -1, axis=1) - jnp.roll(pred, 1, axis=1)
    dx_target = jnp.roll(target, -1, axis=1) - jnp.roll(target, 1, axis=1)
    dy_pred = jnp.roll(pred, -1, axis=2) - jnp.roll(pred, 1, axis=2)
    dy_target = jnp.roll(target, -1, axis=2) - jnp.roll(target, 1, axis=2)

    dx_diff_flat = (dx_pred - dx_target).reshape(pred.shape[0], -1)
    dx_target_flat = dx_target.reshape(pred.shape[0], -1)
    dy_diff_flat = (dy_pred - dy_target).reshape(pred.shape[0], -1)
    dy_target_flat = dy_target.reshape(pred.shape[0], -1)

    diff_dx = jnp.sum(dx_diff_flat ** 2, axis=-1)
    ynorm_dx = jnp.sum(dx_target_flat ** 2, axis=-1)
    diff_dy = jnp.sum(dy_diff_flat ** 2, axis=-1)
    ynorm_dy = jnp.sum(dy_target_flat ** 2, axis=-1)

    # Total H1 norm: value + gradients
    diff_total = diff_l2 + diff_dx + diff_dy
    ynorm_total = ynorm_l2 + ynorm_dx + ynorm_dy

    # Relative norm per sample, then sum (reference default reduction)
    h1_per_sample = jnp.sqrt(diff_total) / (jnp.sqrt(ynorm_total) + eps)
    return jnp.sum(h1_per_sample)
