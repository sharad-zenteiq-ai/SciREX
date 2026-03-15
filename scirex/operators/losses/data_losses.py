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

In standard machine learning, Absolute Error (like MSE) is often sufficient. 
However, in operator learning (like solving PDEs), different samples can have 
vastly different scales. We use relative loss functions (Lp, H1) to ensure 
the model learns equally from all samples, regardless of their absolute magnitude.
"""
import jax.numpy as jnp

def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    Mean Squared Error.
    
    Mathematical Formula:
    $\mathcal{L}_{MSE} = \frac{1}{N} \sum (\hat{u} - u)^2$
    
    Calculates the absolute squared differences across all points.
    """
    return jnp.mean((pred - target) ** 2)


def lp_loss(pred: jnp.ndarray, target: jnp.ndarray, p: int = 2) -> jnp.ndarray:
    """
    Relative Lp loss (Default is L2).
    Standard objective for operator learning.
    
    Mathematical Formula:
    $\mathcal{L}_{L_2} = \frac{1}{N} \sum_{i=1}^{N} \frac{\| \hat{u}^{(i)} - u^{(i)} \|_p}{\| u^{(i)} \|_p + \epsilon}$
    
    This measures the magnitude of the error scaled by the magnitude of the true target.
    """
    # 1. Flatten spatial and channel dimensions to (batch, -1)
    # This turns multi-dimensional grids into flat 1D arrays for each sample.
    batch = pred.shape[0]
    diff = (pred - target).reshape(batch, -1)
    targ = target.reshape(batch, -1)
    
    # 2. Compute norms per sample in batch
    # Calculates the Euclidean distance (L2 norm) for the error and the target.
    # axis=1 ensures we calculate this per sample, independently.
    diff_norm = jnp.linalg.norm(diff, ord=p, axis=1)
    targ_norm = jnp.linalg.norm(targ, ord=p, axis=1)
    
    # 3. Return mean relative error over batch
    # Divides the error magnitude by the target magnitude for each sample, 
    # then averages those percentages across the entire batch.
    return jnp.mean(diff_norm / (targ_norm + 1e-8))


def h1_loss(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """
    Relative H1 Sobolev loss.
    
    Mathematical Formula:
    $\mathcal{L}_{H_1} = \sum_{i=1}^{N} \frac{\sqrt{ \| \hat{u}^{(i)} - u^{(i)} \|_2^2 + \| \nabla_x (\hat{u}^{(i)} - u^{(i)}) \|_2^2 + \| \nabla_y (\hat{u}^{(i)} - u^{(i)}) \|_2^2 }}{\sqrt{ \| u^{(i)} \|_2^2 + \| \nabla_x u^{(i)} \|_2^2 + \| \nabla_y u^{(i)} \|_2^2 } + \epsilon}$

    This loss forces the network to match both the raw values AND the first-order 
    spatial derivatives (the slopes/gradients). It heavily penalizes predictions 
    that are physically jagged or inconsistent.

    Args:
        pred: Predictions, shape ``(batch, nx, ny, ...)``.
        target: Ground truth, same shape as *pred*.
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar H1 loss.
    """
    
    # ==================================================================
    # Phase 1: The Raw Value (L2) Error
    # ==================================================================
    # Flatten the grids into flat arrays to compute the base squared errors.
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    # Sum of squared differences for the raw values
    diff_l2 = jnp.sum((pred_flat - target_flat) ** 2, axis=-1)
    
    # Sum of squares of the true target values
    ynorm_l2 = jnp.sum(target_flat ** 2, axis=-1)

    # ==================================================================
    # Phase 2 & 3: Spatial Derivatives via Periodic Central Differences
    # ==================================================================
    # Using jnp.roll to shift the grid left/right/up/down. 
    # Subtracting the shifts gives us the slope/gradient at every point.
    
    # X-Gradient (axis 1: e.g., Vertical Slope)
    dx_pred = jnp.roll(pred, -1, axis=1) - jnp.roll(pred, 1, axis=1)
    dx_target = jnp.roll(target, -1, axis=1) - jnp.roll(target, 1, axis=1)
    
    # Y-Gradient (axis 2: e.g., Horizontal Slope)
    dy_pred = jnp.roll(pred, -1, axis=2) - jnp.roll(pred, 1, axis=2)
    dy_target = jnp.roll(target, -1, axis=2) - jnp.roll(target, 1, axis=2)

    # Flatten the newly calculated gradient grids
    dx_diff_flat = (dx_pred - dx_target).reshape(pred.shape[0], -1)
    dx_target_flat = dx_target.reshape(pred.shape[0], -1)
    dy_diff_flat = (dy_pred - dy_target).reshape(pred.shape[0], -1)
    dy_target_flat = dy_target.reshape(pred.shape[0], -1)

    # Sum of squared differences for the X and Y gradients
    diff_dx = jnp.sum(dx_diff_flat ** 2, axis=-1)
    ynorm_dx = jnp.sum(dx_target_flat ** 2, axis=-1)
    
    diff_dy = jnp.sum(dy_diff_flat ** 2, axis=-1)
    ynorm_dy = jnp.sum(dy_target_flat ** 2, axis=-1)

    # ==================================================================
    # Phase 4: Final Sobolev Consolidation
    # ==================================================================
    # Add value errors + vertical slope errors + horizontal slope errors
    diff_total = diff_l2 + diff_dx + diff_dy
    
    # Add true value scale + true vertical scale + true horizontal scale
    ynorm_total = ynorm_l2 + ynorm_dx + ynorm_dy

    # Compute the relative norm per sample (square roots applied here)
    h1_per_sample = jnp.sqrt(diff_total) / (jnp.sqrt(ynorm_total) + eps)
    
    # Return the sum over the batch (reference default reduction)
    return jnp.sum(h1_per_sample)
