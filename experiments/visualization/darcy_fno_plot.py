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

"""Train FNO2D and plot results (prediction vs truth) for the 2D Darcy problem."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scirex.operators.models.fno2d import FNO2D
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse
from scirex.data.datasets.darcy import generator as darcy_generator

def main():
    # Config
    batch_size = 16
    resolution = (64, 64)
    in_channels = 1
    hidden_channels = 32  # Smaller width for faster execution in plotting demo
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

    # Training generator - Using FNO-style binary permeability
    num_batches = 500
    gen = darcy_generator(
        num_batches=num_batches, 
        batch_size=batch_size, 
        nx=nx, ny=ny, 
        rng_seed=0,
        mode="binary",
        a_low=4.0,
        a_high=12.0
    )

    losses = []
    print(f"Starting training (FNO Darcy plotting, n_modes={n_modes})...")
    for step, (a_np, u_np) in enumerate(gen):
        batch = {"x": jnp.asarray(a_np), "y": jnp.asarray(u_np)}
        state, metrics = train_step(state, batch, mse)
        losses.append(float(metrics["loss"]))
        if step % 20 == 0:
            print(f"step {step:4d}, loss: {losses[-1]:.6e}")

    # Inference on a fresh test sample
    a_test, u_test = next(darcy_generator(
        num_batches=1, batch_size=1, nx=nx, ny=ny, 
        rng_seed=999, mode="binary", a_low=4.0, a_high=12.0
    ))
    eval_batch = {"x": jnp.asarray(a_test), "y": jnp.asarray(u_test)}
    out = eval_step(state, eval_batch, mse)
    u_pred = np.array(out["preds"][0, ..., 0])
    u_true = u_test[0, ..., 0]
    a_field = a_test[0, ..., 0]

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im0 = axes[0, 0].imshow(a_field, origin='lower')
    axes[0, 0].set_title("Input Permeability a(x,y)")
    fig.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(u_true, origin='lower')
    axes[0, 1].set_title("True Pressure u(x,y)")
    fig.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(u_pred, origin='lower')
    axes[1, 0].set_title(f"FNO prediction (n_modes={n_modes})")
    fig.colorbar(im2, ax=axes[1, 0])
    
    err = np.abs(u_true - u_pred)
    im3 = axes[1, 1].imshow(err, origin='lower', cmap='inferno')
    axes[1, 1].set_title("Absolute error |u - u_pred|")
    fig.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    out_dir = "experiments/results/figures"
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "darcy_fno_results.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")

    # Secondary plot for loss
    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.title("Darcy FNO Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "darcy_fno_loss.png"), dpi=150)
    print(f"Saved loss plot to {os.path.join(out_dir, 'darcy_fno_loss.png')}")
    plt.close()

if __name__ == "__main__":
    main()
