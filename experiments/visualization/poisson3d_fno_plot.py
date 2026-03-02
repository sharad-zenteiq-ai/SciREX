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
Visualize FNO 3D Poisson Results.
Loads best checkpoint and plots Ground Truth vs Prediction slices.
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

from scirex.operators.models.fno import FNO3D
from scirex.operators.data import random_poisson_3d_batch
from configs.poisson_fno_config import FNO3DConfig

class UnitGaussianNormalizer:
    """Normalizer for 3D spatial fields."""
    def __init__(self, x_ref, eps=1e-5):
        # x shape: (batch, nx, ny, nz, channels)
        self.mean = jnp.mean(x_ref, axis=(0, 1, 2, 3), keepdims=True)
        self.std = jnp.std(x_ref, axis=(0, 1, 2, 3), keepdims=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

def plot_3d_slice_comparison(f_test, u_test, u_pred, results_dir, filename="poisson3d_slices.png"):
    """
    Plot 4 columns: Source Term, True Solution, FNO Prediction, Absolute Error.
    Each row is a different sample at the middle-Z slice.
    """
    num_samples = min(3, f_test.shape[0])
    nx, ny, nz = u_test.shape[1], u_test.shape[2], u_test.shape[3]
    z_slice = nz // 2
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    
    for i in range(num_samples):
        # 1. Source Term f (middle-Z slice)
        ax = axes[i, 0]
        # input is (B, nx, ny, nz, 4) if include_mesh=True. Channel 0 is Source.
        im = ax.imshow(f_test[i, :, :, z_slice, 0], cmap='viridis')
        ax.set_title(f"Source f (Z={z_slice})")
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        
        # 2. True Solution u (middle-Z slice)
        ax = axes[i, 1]
        im = ax.imshow(u_test[i, :, :, z_slice, 0], cmap='jet')
        ax.set_title("True Solution u")
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        
        # 3. FNO Prediction u_pred (middle-Z slice)
        ax = axes[i, 2]
        im = ax.imshow(u_pred[i, :, :, z_slice, 0], cmap='jet')
        ax.set_title("FNO Prediction")
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        
        # 4. Absolute Error (middle-Z slice)
        ax = axes[i, 3]
        error = np.abs(u_test[i, :, :, z_slice, 0] - u_pred[i, :, :, z_slice, 0])
        rel_err = np.linalg.norm(error) / (np.linalg.norm(u_test[i, ..., 0]) + 1e-8)
        im = ax.imshow(error, cmap='hot')
        ax.set_title(f"Abs Error (Rel: {rel_err:.2%})")
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Slice comparison saved to {save_path}")

def plot_3d_volume_realization(u_test_sample, u_pred_sample, results_dir, filename="poisson3d_volume.png"):
    """
    Plot 3D volume visualization for a single sample.
    """
    fig = plt.figure(figsize=(15, 7))
    
    # 1. Ground Truth Volume
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # Use thresholding to show high-magnitude structure
    threshold = u_test_sample.mean() + 0.5 * u_test_sample.std()
    voxels = u_test_sample > threshold
    ax1.voxels(voxels, facecolors='cyan', edgecolor='k', alpha=0.3)
    ax1.set_title("Target 3D Structure")
    
    # 2. Prediction Volume
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    voxels_pred = u_pred_sample > threshold
    ax2.voxels(voxels_pred, facecolors='orange', edgecolor='k', alpha=0.3)
    ax2.set_title("Pred 3D Structure")
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Volume visualization saved to {save_path}")

def main():
    # 1. Load Configuration
    config = FNO3DConfig()
    
    # 2. Initialize Model (Consistent with training preset)
    print(f"Initializing FNO3D model (hidden_channels={config.hidden_channels})...")
    model = FNO3D(
        hidden_channels=config.hidden_channels, 
        n_layers=config.n_layers, 
        n_modes=config.n_modes, 
        out_channels=config.out_channels,
        lifting_channel_ratio=config.lifting_channel_ratio,
        projection_channel_ratio=config.projection_channel_ratio,
        use_grid=config.use_grid,
        use_norm=config.use_norm,
        fno_skip=config.fno_skip,
        channel_mlp_skip=config.channel_mlp_skip,
        use_channel_mlp=config.use_channel_mlp,
        padding=config.domain_padding,
        activation=nn.gelu
    )
    
    # 3. Load Checkpoint
    # Default to the lploss specific weight if it exists
    lploss_ckpt = os.path.join(project_root, "experiments/checkpoints/poisson3d_fno_lploss_params.pkl")
    standard_ckpt = os.path.join(project_root, "experiments/checkpoints/poisson3d_fno_params.pkl")
    
    if os.path.exists(lploss_ckpt):
        ckpt_path = lploss_ckpt
    elif os.path.exists(standard_ckpt):
        ckpt_path = standard_ckpt
    else:
        print(f"Error: No checkpoint found at {lploss_ckpt} or {standard_ckpt}")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    
    nx, ny, nz = config.resolution
    dummy_input = jnp.ones((1, nx, ny, nz, config.in_channels))
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
    print("Fitting normalizers and generating test batch...")
    # Use config seed for reference batch to match training stats
    f_train_ref, u_train_ref = random_poisson_3d_batch(
        batch_size=50, nx=nx, ny=ny, nz=nz, channels=1, 
        rng_seed=config.seed, include_mesh=True
    )
    x_normalizer = UnitGaussianNormalizer(f_train_ref)
    y_normalizer = UnitGaussianNormalizer(u_train_ref)

    # Generate Test Data
    f_test, u_test = random_poisson_3d_batch(
        batch_size=5, nx=nx, ny=ny, nz=nz, channels=1, 
        rng_seed=999, include_mesh=True
    )
    
    # 5. Inference
    print("Running inference...")
    f_test_encoded = x_normalizer.encode(jnp.asarray(f_test))
    u_pred_encoded = model.apply({"params": loaded_params}, f_test_encoded)
    u_pred = y_normalizer.decode(u_pred_encoded)
    
    # 6. Visualization
    results_dir = os.path.join(project_root, "experiments/results/poisson3d")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot Slices (Z-middle)
    plot_3d_slice_comparison(f_test, u_test, u_pred, results_dir)
    
    # Plot Volume for the first sample
    plot_3d_volume_realization(np.array(u_test[0, ..., 0]), np.array(u_pred[0, ..., 0]), results_dir)
    
    # 7. Check for Loss History
    metrics_path = os.path.join(results_dir, "fno3d_metrics.json")
    if os.path.exists(metrics_path):
        print(f"Plotting loss history from {metrics_path}...")
        with open(metrics_path, "r") as f:
            history = json.load(f)
        
        epochs_range = range(len(history["train_rel_l2"]))
        plt.figure(figsize=(10, 6))
        plt.semilogy(epochs_range, history["train_rel_l2"], label='Train Rel L2')
        plt.semilogy(epochs_range, history["test_rel_l2"], label='Test Rel L2', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Error')
        plt.title('3D Poisson FNO: Loss Convergence')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(results_dir, "poisson3d_loss_plot.png"), dpi=150)
        plt.close()

    print(f"Visualizations completed. Check outputs in: {results_dir}")

if __name__ == "__main__":
    main()
