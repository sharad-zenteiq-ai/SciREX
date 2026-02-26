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

"""Train a Flax FNO to learn the 2D Darcy solution operator (a -> u)."""
import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.models.fno2d import FNO2D
from scirex.operators.layers import Lifting, Projection
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse
from scirex.data.datasets.darcy import generator as darcy_generator

def main():
    # Model Config
    batch_size = 16
    resolution = (64, 64)
    in_channels = 1
    hidden_channels = 64
    n_layers = 4
    n_modes = (16, 16)
    out_channels = 1
    lr = 1e-3
    rng = jax.random.PRNGKey(42)

    model = FNO2D(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        n_modes=n_modes, 
        out_channels=out_channels
    )
    nx, ny = resolution
    state = create_train_state(rng, model, (batch_size, nx, ny, in_channels), learning_rate=lr)

    # Dataset config - Using FNO-style binary permeability {4, 12}
    num_batches = 500
    gen = darcy_generator(
        num_batches=num_batches, 
        batch_size=batch_size, 
        nx=nx, ny=ny, 
        rng_seed=0,
        mode="binary",  # FNO-style thresholded GRF
        a_low=4.0,
        a_high=12.0
    )

    nx, ny = resolution
    print(f"Starting training (FNO 2D Darcy, n_modes={n_modes})...")
    for step, (a_np, u_np) in enumerate(gen):
        batch = {"x": jnp.asarray(a_np), "y": jnp.asarray(u_np)}
        state, metrics = train_step(state, batch, mse)
        
        if step % 10 == 0:
            print(f"step {step:4d}, loss: {float(metrics['loss']):.6e}")
            
        if step == num_batches - 1 or (step > 0 and step % 50 == 0):
            eval_metrics = eval_step(state, batch, mse)
            print(f"  eval loss at step {step}: {float(eval_metrics['loss']):.6e}")

    # Final validation on fresh data
    print("Final evaluation...")
    nx, ny = resolution
    a_test, u_test = next(darcy_generator(
        num_batches=1, batch_size=batch_size, nx=nx, ny=ny, 
        rng_seed=999, mode="binary", a_low=4.0, a_high=12.0
    ))
    eval_batch = {"x": jnp.asarray(a_test), "y": jnp.asarray(u_test)}
    final_metrics = eval_step(state, eval_batch, mse)
    print("Final eval loss:", float(final_metrics["loss"]))

if __name__ == "__main__":
    main()
