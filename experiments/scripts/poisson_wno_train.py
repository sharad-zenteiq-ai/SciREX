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

"""Train a Flax WNO to learn the Poisson solution operator (f -> u)."""
import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.models.wno2d import WNO2D
from scirex.operators.layers import Lifting, Projection
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse
from scirex.data.datasets.poisson import generator as poisson_generator

def main():
    batch_size = 32
    nx, ny, in_channels = 64, 64, 1
    hidden_channels = 64
    n_layers = 4
    levels = 1
    wavelet = "db4"  # Using Daubechies 4 for better frequency localization
    out_channels = 1
    lr = 1e-3
    rng = jax.random.PRNGKey(42)

    model = WNO2D(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        levels=levels, 
        wavelet=wavelet, 
        out_channels=out_channels
    )
    state = create_train_state(rng, model, (batch_size, nx, ny, in_channels), learning_rate=lr)

    num_batches = 500
    gen = poisson_generator(num_batches=num_batches, batch_size=batch_size, nx=nx, ny=ny, channels=in_channels, rng_seed=0)

    print(f"Starting training (WNO 2D Poisson with {wavelet})...")
    for step, (f_np, u_np) in enumerate(gen):
        batch = {"x": jnp.asarray(f_np), "y": jnp.asarray(u_np)}
        state, metrics = train_step(state, batch, mse)
        if step % 100 == 0:
            print(f"step {step:4d}, loss: {float(metrics['loss']):.6e}")
        if step == num_batches - 1 or step % 500 == 0:
            eval_metrics = eval_step(state, batch, mse)
            print(f"  eval loss: {float(eval_metrics['loss']):.6e}")

    f_test, u_test = next(poisson_generator(num_batches=1, batch_size=batch_size, nx=nx, ny=ny, channels=in_channels, rng_seed=999))
    eval_batch = {"x": jnp.asarray(f_test), "y": jnp.asarray(u_test)}
    final_metrics = eval_step(state, eval_batch, mse)
    print("Final eval loss:", float(final_metrics["loss"]))

if __name__ == "__main__":
    main()
