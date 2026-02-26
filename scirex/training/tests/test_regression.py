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
import optax
from scirex.operators.models.fno2d import FNO2D
from scirex.training.train_state import create_train_state

def test_fno2d_overfit_toy():
    """Regression test: Check if FNO2D can overfit a single toy sample."""
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_channels = 1, 16, 16, 1
    hidden_channels, n_layers = 16, 2
    n_modes = (4, 4)
    out_channels = 1
    
    model = FNO2D(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        n_modes=n_modes, 
        out_channels=out_channels,
        activation=lambda x: x
    )
    
    # Toy data: identity mapping-like
    x = jax.random.normal(rng, (batch, nx, ny, in_channels))
    y_target = x * 2.0 # Simple target
    
    state = create_train_state(rng, model, (batch, nx, ny, in_channels), learning_rate=1e-2)
    
    @jax.jit
    def train_step(state, x, y):
        def loss_fn(params):
            preds = state.apply_fn({"params": params}, x)
            return jnp.mean((preds - y) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    initial_loss = 0
    for i in range(100):
        state, loss = train_step(state, x, y_target)
        if i == 0:
            initial_loss = loss
            
    final_loss = loss
    print(f"Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")
    assert final_loss < initial_loss * 0.1 # Should decrease significantly
