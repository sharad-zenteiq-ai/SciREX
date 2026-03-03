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
from flax import linen as nn
import json

from scirex.operators.models.fno import FNO2D
from scirex.operators.training import create_train_state
from scirex.operators.data import generator as poisson_generator
from configs.poisson_fno_config import FNO2DConfig

class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        self.mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
        self.std = jnp.std(x, axis=(0, 1, 2), keepdims=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

def main():
    # 1. Load Configuration
    config = FNO2DConfig()
    
    # 2. Initialize Model (Consistent with training preset)
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
        channel_mlp_skip=config.channel_mlp_skip,
        use_channel_mlp=config.use_channel_mlp,
        padding=config.domain_padding,
        activation=nn.gelu
    )
    
    # 3. Load Checkpoint
    ckpt_path = os.path.join(project_root, "experiments/checkpoints/poisson_fno_params.pkl")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    
    nx, ny = config.resolution
    dummy_input = jnp.ones((1, nx, ny, config.in_channels))
    variables = model.init(jax.random.PRNGKey(0), dummy_input)
    init_params = variables["params"]
    
    with open(ckpt_path, "rb") as f:
        ckpt_bytes = f.read()
    
    try:
        loaded_params = flax.serialization.from_bytes(init_params, ckpt_bytes)
        print("Successfully loaded parameters.")
    except Exception as e:
        print(f"Failed to load parameters: {e}")
        return

    # 4. Generate Data and Recreate Normalizers
    print("Generating validation batch and fitting normalizers...")
    from scirex.operators.data import random_poisson_batch
    
    # Re-generate train sample (same seed as training) to get correct scales
    f_train_ref, u_train_ref = random_poisson_batch(
        batch_size=100, nx=nx, ny=ny, channels=1, rng_seed=config.seed
    )
    x_normalizer = UnitGaussianNormalizer(f_train_ref)
    y_normalizer = UnitGaussianNormalizer(u_train_ref)

    # Generate Test Data (different seed)
    f_test, u_test = random_poisson_batch(
        batch_size=8, nx=nx, ny=ny, channels=1, rng_seed=12345
    )
    
    # 5. Inference
    print("Running inference...")
    f_test_encoded = x_normalizer.encode(jnp.asarray(f_test))
    u_pred_encoded = model.apply({"params": loaded_params}, f_test_encoded)
    u_pred = y_normalizer.decode(u_pred_encoded)
    
    # 6. Plotting Field Comparisons (Only 2D Squares)
    plot_dir = os.path.join(project_root, "experiments/results/poisson")
    os.makedirs(plot_dir, exist_ok=True)
    
    num_plots = min(3, f_test.shape[0])
    fig, axes = plt.subplots(num_plots, 4, figsize=(16, 4 * num_plots))
    
    for i in range(num_plots):
        ax_row = axes[i] if num_plots > 1 else axes
            
        # Input f
        ax = ax_row[0]
        im = ax.imshow(f_test[i, ..., 0], cmap="viridis")
        ax.set_title(f"Source Term f")
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        
        # Ground Truth u
        ax = ax_row[1]
        im = ax.imshow(u_test[i, ..., 0], cmap="inferno")
        ax.set_title("True Solution u")
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        
        # Prediction u_pred
        ax = ax_row[2]
        im = ax.imshow(u_pred[i, ..., 0], cmap="inferno")
        ax.set_title("FNO Prediction")
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        
        # Error
        ax = ax_row[3]
        err = np.abs(u_test[i, ..., 0] - u_pred[i, ..., 0])
        rel_err = np.linalg.norm(err) / np.linalg.norm(u_test[i, ..., 0])
        im = ax.imshow(err, cmap="magma")
        ax.set_title(f"Error (Rel: {rel_err:.2%})")
        fig.colorbar(im, ax=ax)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(plot_dir, "poisson_fno_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"Comparison plots saved to {save_path}")
    print("Loss plotting skipped as requested.")

if __name__ == "__main__":
    main()
