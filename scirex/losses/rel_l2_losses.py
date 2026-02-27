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
from scirex.losses.data_losses import lp_loss

def phys_rel_l2_loss(pred_encoded: jnp.ndarray, target_encoded: jnp.ndarray, normalizer) -> jnp.ndarray:
    """
    Computes the Relative L2 loss in the original physical (unnormalized) domain.
    
    This function leverages a provided normalizer to decode the model outputs 
    back to their true physical scales before computing the relative error. 
    This ensures the loss reflects the actual physical accuracy of the prediction 
    rather than its performance on a normalized latent representation.

    Args:
        pred_encoded (jnp.ndarray): The raw, potentially normalized, model prediction.
        target_encoded (jnp.ndarray): The raw, potentially normalized, target field.
        normalizer: An object (e.g., UnitGaussianNormalizer) implementing .decode().
        
    Returns:
        jnp.ndarray: The Relative L2 loss (scalar).
    """
    # 1. Back-transform outputs to original unnormalized PDE scales
    pred_decoded = normalizer.decode(pred_encoded)
    target_decoded = normalizer.decode(target_encoded)

    # 2. Compute the lp_loss (Relative L2 norm) over the physical signals
    return lp_loss(pred_decoded, target_decoded, p=2)
