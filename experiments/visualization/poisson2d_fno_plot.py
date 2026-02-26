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
Visualize FNO 2D Poisson Results using trained weights.
Loads the best checkpoint and plots Ground Truth vs Prediction.
"""
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import flax
import json

from scirex.operators.models.fno2d import FNO2D
from scirex.training.train_state import create_train_state
from scirex.data.datasets.poisson import generator as poisson_generator
from configs.poisson_fno_config import FNO2DConfig

def main():
    # 1. Load Configuration
    config = FNO2DConfig()
    
    # 2. Initialize Model (Same architecture as training)
    print(f"Initializing FNO2D model...")
    model = FNO2D(
        hidden_channels=config.hidden_channels, 
        n_layers=config.n_layers, 
        n_modes=config.n_modes, 
        out_channels=config.out_channels,
        lifting_channel_ratio=config.lifting_channel_ratio,
        projection_channel_ratio=config.projection_channel_ratio,
        use_grid=config.use_grid,
        fno_skip=config.fno_skip,
        use_channel_mlp=config.use_channel_mlp
    )
    
    # 3. Load Checkpoint
    ckpt_path = os.path.join(project_root, "experiments/checkpoints/poisson_fno_params.pkl")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Please run experiments/scripts/poisson_fno_train.py first.")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    
    # Create dummy state to initialize structure
    nx, ny = config.resolution
    dummy_input = jnp.ones((1, nx, ny, config.in_channels))
    variables = model.init(jax.random.PRNGKey(0), dummy_input)
    init_params = variables["params"]
    
    # Load bytes and restore
    with open(ckpt_path, "rb") as f:
        ckpt_bytes = f.read()
    
    # Use flax serialization to restore params
    try:
        loaded_params = flax.serialization.from_bytes(init_params, ckpt_bytes)
        print("Successfully loaded parameters.")
    except Exception as e:
        print(f"Failed to load parameters: {e}")
        return

    # 4. Generate Test Data
    print("Generating validation batch...")
    # Use a specific seed to see new unseen data
    # Note: poisson_generator yields (f, u) batches
    nx, ny = config.resolution
    gen = poisson_generator(
        num_batches=1, 
        batch_size=8, # visualize enough examples
        nx=nx, 
        ny=ny, 
        channels=config.in_channels, 
        rng_seed=12345 
    )
    
    f_test, u_test = next(gen)
    
    # 5. Inference
    print("Running inference...")
    u_pred = model.apply({"params": loaded_params}, jnp.asarray(f_test))
    
    # 6. Plotting Predictions
    # Ensure plot directory exists
    plot_dir = os.path.join(project_root, "experiments/results/poisson")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    num_plots = min(3, f_test.shape[0])
    fig, axes = plt.subplots(num_plots, 4, figsize=(16, 4 * num_plots))
    # Columns: Input (f), Ground Truth (u), Prediction (u_hat), Error |u - u_hat|
    
    for i in range(num_plots):
        # Input f
        if num_plots > 1:
            ax_row = axes[i]
        else:
            ax_row = axes
            
        ax = ax_row[0]
        im = ax.imshow(f_test[i, ..., 0], cmap="viridis")
        ax.set_title(f"Input f (Sample {i})")
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        
        # Ground Truth u
        ax = ax_row[1]
        im = ax.imshow(u_test[i, ..., 0], cmap="inferno")
        ax.set_title("Ground Truth u")
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        
        # Prediction u_pred
        ax = ax_row[2]
        im = ax.imshow(u_pred[i, ..., 0], cmap="inferno")
        ax.set_title("Prediction u_pred")
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        
        # Error
        ax = ax_row[3]
        err = np.abs(u_test[i, ..., 0] - u_pred[i, ..., 0])
        max_err = np.max(err)
        im = ax.imshow(err, cmap="magma")
        ax.set_title(f"Abs Error (Max: {max_err:.4e})")
        fig.colorbar(im, ax=ax)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(plot_dir, "poisson_fno_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"Prediction plots saved to {save_path}")

    # 7. Plot Loss Curves (if available)
    metrics_path = os.path.join(plot_dir, "fno2d_metrics.json")
    if os.path.exists(metrics_path):
        print(f"Loading training metrics from {metrics_path}...")
        with open(metrics_path, "r") as f:
            history = json.load(f)
        
        epochs_range = range(len(history["train_mse"]))
        plt.figure(figsize=(12, 5))
        
        # Plot MSE
        plt.subplot(1, 2, 1)
        plt.semilogy(epochs_range, history["train_mse"], label='Train MSE')
        if "test_mse" in history:
            plt.semilogy(epochs_range, history["test_mse"], label='Test MSE', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Loss (MSE)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        # Plot Rel L2
        plt.subplot(1, 2, 2)
        if "train_rel_l2" in history:
            plt.semilogy(epochs_range, history["train_rel_l2"], label='Train Rel L2')
        plt.semilogy(epochs_range, history["test_rel_l2"], label='Test Rel L2', color='orange', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Error')
        plt.title('Accuracy (Rel L2)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        loss_plot_path = os.path.join(plot_dir, "poisson_fno_losses.png")
        plt.savefig(loss_plot_path, dpi=150)
        print(f"Loss curves saved to: {loss_plot_path}")
    else:
        print(f"Warning: Metrics file not found at {metrics_path}. Skipping loss plots.")

if __name__ == "__main__":
    main()
