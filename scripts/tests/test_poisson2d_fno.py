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
from scirex.operators.models.fno import FNO2D
from scirex.operators.training.train_state import create_train_state
from scirex.operators.training.step_fns import train_step
from scirex.operators.losses.data_losses import lp_loss
from scirex.operators.data.poisson import random_poisson_2d_batch

def test_fno2d_poisson():
    """
    End-to-end integration test: verifies if FNO2D can learn 
    from a small batch of 2D Poisson data using the standard pipeline.
    """
    rng = jax.random.PRNGKey(42)
    rng_init, rng_data = jax.random.split(rng)
    
    # Hyperparameters for a tiny test run
    batch_size = 4
    nx, ny = 16, 16
    n_modes = (4, 4)
    hidden_channels = 16
    n_layers = 2
    learning_rate = 1e-3
    in_channels = 3 # 1 (f) + 2 (grid)
    
    # 1. Generate realistic Poisson data
    x_batch, y_batch = random_poisson_2d_batch(
        batch_size=batch_size, nx=nx, ny=ny, rng_seed=42
    )
    batch = {
        "x": jnp.asarray(x_batch),
        "y": jnp.asarray(y_batch)
    }
    
    # 2. Initialize Model
    model = FNO2D(
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_modes=n_modes,
        out_channels=1,
        use_grid=False # Data already has grid
    )
    
    state = create_train_state(
        rng_init, model, (batch_size, nx, ny, in_channels), learning_rate
    )
    
    # 3. Training Loop
    def loss_fn(pred, target):
        return lp_loss(pred, target, p=2)
    
    initial_loss = float(train_step(state, batch, loss_fn)[1]["loss"])
    
    # Optimized steps to ensure reduction
    for _ in range(100):
        state, metrics = train_step(state, batch, loss_fn)
        
    final_loss = float(metrics["loss"])
    
    print(f"Initial Relative L2: {initial_loss:.6f}, Final Relative L2: {final_loss:.6f}")
    
    # Sanity check: Loss should decrease as it overfits the tiny batch
    assert final_loss < initial_loss
    # It should reach some decent level of fitting even in 100 steps
    assert final_loss < 0.5