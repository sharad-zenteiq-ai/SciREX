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

from typing import Any, Optional
import jax
import optax
from flax.training import train_state

# Alias for type hinting
TrainState = train_state.TrainState

def create_train_state(
    rng: Any, 
    model: Any, 
    input_shape: tuple, 
    learning_rate: Optional[float] = None, 
    weight_decay: float = 1e-4,
    tx: Optional[optax.GradientTransformation] = None
) -> TrainState:
    """
    Initializes the model parameters and constructs the training state.
    
    This factory function handles the initialization of model weights using a 
    dummy input and sets up a standard AdamW optimizer with gradient 
    clipping to ensure stable training of neural operators.

    Args:
        rng (Any): JAX PRNG key for parameter initialization.
        model (Any): Flax Module instance (e.g., FNO2D or WNO2D).
        input_shape (tuple): Expected input shape (batch, ..., channels) for initialization.
        learning_rate (float, optional): Learning rate for the AdamW optimizer.
        weight_decay (float): L2 regularization factor.
        tx (optax.GradientTransformation, optional): Custom optimizer chain. 

    Returns:
        TrainState: An initialized Flax TrainState object.
    """
    # 1. Parameter initialization via a dummy forward pass
    dummy = jax.random.normal(rng, input_shape)
    variables = model.init(rng, dummy)
    params = variables["params"]
    
    # 2. Optimizer configuration
    if tx is None:
        if learning_rate is None:
            raise ValueError("A learning_rate must be provided if no custom optimizer (tx) is specified.")
        
        # Standard configuration for Operator Learning: AdamW + Global Gradient Norm Clipping
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate, weight_decay=weight_decay)
        )
        
    # 3. Create consolidated state
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state
