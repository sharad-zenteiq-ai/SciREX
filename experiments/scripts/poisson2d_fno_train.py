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

"""Train a Flax FNO to learn the Poisson solution operator (f -> u)."""
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import time
import matplotlib.pyplot as plt
import json

from scirex.operators.models.fno2d import FNO2D
from scirex.training.train_state import create_train_state, TrainState
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse, lp_loss
from scirex.data.datasets.poisson import generator as poisson_generator
from configs.poisson_fno_config import FNO2DConfig


def make_schedule(config: FNO2DConfig):
    """Create learning rate schedule (StepLR equivalent)."""
    # boundaries are in steps: epoch * steps_per_epoch
    # values are scaled by gamma cumulatively
    scales = {}
    decay_steps = config.scheduler_step_size * config.steps_per_epoch
    
    # We will compute schedule for total epochs
    num_decays = config.epochs // config.scheduler_step_size
    
    current_scale = 1.0
    for i in range(1, num_decays + 1):
        boundary = i * decay_steps
        current_scale *= config.scheduler_gamma
        scales[boundary] = current_scale
        
    # optax piecewise constant schedule uses init_value * scale for the interval
    schedule = optax.piecewise_constant_schedule(
        init_value=config.learning_rate,
        boundaries_and_scales=scales
    )
    return schedule

def main():
    # 1. Load Configuration
    config = FNO2DConfig()
    
    # Prng Key
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    # 2. Initialize Model
    print(f"Initializing FNO2D (hidden_channels={config.hidden_channels}, modes={config.n_modes})...")
    model = FNO2D(
        hidden_channels=config.hidden_channels, 
        n_layers=config.n_layers, 
        n_modes=config.n_modes, 
        out_channels=config.out_channels
    )
    
    # 3. Initialize Optimizer & Scheduler
    schedule = make_schedule(config)
    # Total steps verification
    total_steps = config.epochs * config.steps_per_epoch
    
    # Create Train State
    # Shape: (batch, nx, ny, in_channels)
    nx, ny = config.resolution
    input_shape = (config.batch_size, nx, ny, config.in_channels)
    state = create_train_state(
        rng=init_rng, 
        model=model, 
        input_shape=input_shape, 
        learning_rate=schedule, 
        weight_decay=config.weight_decay
    )
    
    # 4. Data Generators
    # Train generator yields infinite batches
    nx, ny = config.resolution
    train_gen = poisson_generator(
        num_batches=total_steps + 100, # ensure enough
        batch_size=config.batch_size, 
        nx=nx, 
        ny=ny, 
        channels=config.in_channels, 
        rng_seed=config.seed
    )
    
    # Test set (fixed seed for consistency)
    f_test, u_test = next(poisson_generator(
        num_batches=1, 
        batch_size=config.batch_size, # evaluate on same batch size
        nx=nx, 
        ny=ny, 
        channels=config.in_channels, 
        rng_seed=999 # Different seed for test
    ))
    test_batch = {"x": jnp.asarray(f_test), "y": jnp.asarray(u_test)}
    
    # Create checkpoint directory and results directory
    ckpt_dir = os.path.join(project_root, "experiments/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "poisson_fno_params.pkl")

    results_dir = os.path.join(project_root, "experiments/results/poisson")
    os.makedirs(results_dir, exist_ok=True)

    best_rel_l2 = float("inf")
    
    # Track metrics for plotting
    history = {
        "train_mse": [],
        "train_rel_l2": [],
        "test_mse": [],
        "test_rel_l2": []
    }

    # 5. Training Loop
    print(f"Starting training for {config.epochs} epochs ({total_steps} steps)...")
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for _ in range(config.steps_per_epoch):
            f_np, u_np = next(train_gen)
            batch = {"x": jnp.asarray(f_np), "y": jnp.asarray(u_np)}
            state, metrics = train_step(state, batch, mse)
            epoch_loss += float(metrics["loss"])
            
        epoch_time = time.time() - epoch_start_time
        avg_train_mse = epoch_loss / config.steps_per_epoch
        
        # We need train_rel_l2, test_mse, test_rel_l2
        # For efficiency, compute train_rel_l2 on the LAST training batch of the epoch
        train_metrics_l2 = eval_step(state, batch, lp_loss)
        avg_train_l2 = float(train_metrics_l2["loss"])

        # Evaluate on test set
        test_metrics_mse = eval_step(state, test_batch, mse)
        test_metrics_l2 = eval_step(state, test_batch, lp_loss)
        
        v_test_mse = float(test_metrics_mse["loss"])
        v_test_l2 = float(test_metrics_l2["loss"])

        # Update history
        history["train_mse"].append(avg_train_mse)
        history["train_rel_l2"].append(avg_train_l2)
        history["test_mse"].append(v_test_mse)
        history["test_rel_l2"].append(v_test_l2)
        
        test_rel_l2 = v_test_l2 # for checkpointing
        
        # Save best checkpoint
        if test_rel_l2 < best_rel_l2:
            best_rel_l2 = test_rel_l2
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(state.params))
        
        # Scheduler info
        current_lr = schedule(state.step)
        
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:4d} | "
                  f"Train MSE: {avg_train_mse:.6e} | "
                  f"Test MSE: {v_test_mse:.6e} | "
                  f"Test Rel L2: {v_test_l2:.6f} | "
                  f"Best Rel L2: {best_rel_l2:.6f} | "
                  f"LR: {float(current_lr):.2e} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save history periodically to avoid loss on interruption
            with open(os.path.join(results_dir, "fno2d_metrics.json"), "w") as f:
                json.dump(history, f, indent=4)

    print("\nTraining Complete.")
    print(f"Best Test Relative L2 Error: {best_rel_l2:.6f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # 6. Plot Loss Curves
    epochs_range = range(len(history["train_mse"]))
    plt.figure(figsize=(12, 5))
    
    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.semilogy(epochs_range, history["train_mse"], label='Train MSE')
    plt.semilogy(epochs_range, history["test_mse"], label='Test MSE', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Loss (MSE)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Plot Rel L2
    plt.subplot(1, 2, 2)
    plt.semilogy(epochs_range, history["train_rel_l2"], label='Train Rel L2')
    plt.semilogy(epochs_range, history["test_rel_l2"], label='Test Rel L2', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error')
    plt.title('Accuracy (Rel L2)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    loss_plot_path = os.path.join(results_dir, "poisson_fno_losses.png")
    plt.savefig(loss_plot_path, dpi=150)
    print(f"Loss curves saved to: {loss_plot_path}")

if __name__ == "__main__":
    main()
