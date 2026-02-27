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
    Initialize model params and optimizer state.

    - rng: PRNG key
    - model: Flax Module instance (e.g., FNO2D(...))
    - input_shape: shape tuple for dummy input (batch, nx, ny, channels)
    - learning_rate: optimizer lr (can be float or optax schedule)
    - weight_decay: weight decay for AdamW
    - tx: optional pre-built optax optimizer
    """
    dummy = jax.random.normal(rng, input_shape)
    variables = model.init(rng, dummy)
    params = variables["params"]
    if tx is None:
        if learning_rate is None:
            raise ValueError("Must provide either learning_rate or tx")
        # Using AdamW with gradient clipping (standard for stable training)
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate, weight_decay=weight_decay)
        )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state
