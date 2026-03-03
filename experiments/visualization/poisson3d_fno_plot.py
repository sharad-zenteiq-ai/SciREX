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

def plot_3d_slice_comparison(f_test, u_test, u_pred, results_dir, filename="poisson3d_fno_slices.png"):
    """
    Enhanced Plotting: Multiple slices (Z-mid, Y-mid) with better colormaps.
    Columns: Source, Truth, Prediction, Error.
    """
    num_samples = min(2, f_test.shape[0])
    nx, ny, nz = u_test.shape[1:4]
    z_mid = nz // 2
    y_mid = ny // 2
    
    # Each sample gets 2 rows (Z-slice and Y-slice)
    fig, axes = plt.subplots(num_samples * 2, 4, figsize=(20, 4.5 * num_samples * 2))
    
    for i in range(num_samples):
        # --- Row 2*i: Z-slice ---
        for row_offset, slice_idx, axis_name, slice_plane in [(0, z_mid, 'Z', 'XY'), (1, y_mid, 'Y', 'XZ')]:
            curr_row = 2 * i + row_offset
            
            # 1. Source Term f
            ax = axes[curr_row, 0]
            if slice_plane == 'XY':
                data = f_test[i, :, :, slice_idx, 0]
            else:
                data = f_test[i, :, slice_idx, :, 0]
            im = ax.imshow(data, cmap='viridis', interpolation='nearest')
            ax.set_title(f"Source f ({axis_name}={slice_idx})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
            
            # 2. True Solution u
            ax = axes[curr_row, 1]
            if slice_plane == 'XY':
                data_u = u_test[i, :, :, slice_idx, 0]
            else:
                data_u = u_test[i, :, slice_idx, :, 0]
            im = ax.imshow(data_u, cmap='inferno')
            ax.set_title(f"True u ({slice_plane})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
            
            # 3. FNO Prediction
            ax = axes[curr_row, 2]
            if slice_plane == 'XY':
                data_p = u_pred[i, :, :, slice_idx, 0]
            else:
                data_p = u_pred[i, :, slice_idx, :, 0]
            im = ax.imshow(data_p, cmap='inferno')
            ax.set_title(f"FNO Pred ({slice_plane})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
            
            # 4. Abs Error
            ax = axes[curr_row, 3]
            error_slice = np.abs(data_u - data_p)
            rel_err = np.linalg.norm(error_slice) / (np.linalg.norm(data_u) + 1e-8)
            im = ax.imshow(error_slice, cmap='magma')
            ax.set_title(f"Abs Error (Rel:{rel_err:.2%})")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
            
    plt.tight_layout()
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Enhanced slice comparison saved to {save_path}")

def plot_3d_volume_realization(u_test_sample, u_pred_sample, results_dir, filename="poisson3d_fno_volume.png"):
    """
    Premium cloud-like scatter visualization for 3D scalar fields.
    Now with 3 subplots: True, Predicted, and Absolute Error.
    Using [0, 1] axes and a consistent vibrant gradient colormap.
    """
    fig = plt.figure(figsize=(24, 7))
    
    u_error = np.abs(u_test_sample - u_pred_sample)
    
    # Define common scale for True and Pred for better comparison
    v_min_global = min(u_test_sample.min(), u_pred_sample.min())
    v_max_global = max(u_test_sample.max(), u_pred_sample.max())

    def add_cloud_subplot(ax, data, title, vmin=None, vmax=None, cmap_name='jet'):
        dx, dy, dz = data.shape
        # Use [0, 1] coordinates as per data generation
        x_lin = np.linspace(0, 1, dx)
        y_lin = np.linspace(0, 1, dy)
        z_lin = np.linspace(0, 1, dz)
        x, y, z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
        
        vals = data.flatten()
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        v_min = vmin if vmin is not None else vals.min()
        v_max = vmax if vmax is not None else vals.max()
        v_norm = (vals - v_min) / (v_max - v_min + 1e-8)
        
        # Show points with intensity > 15% (normalized)
        mask = v_norm > 0.15
        p_filtered = points[mask]
        v_filtered = v_norm[mask]
        actual_vals_filtered = vals[mask]
        
        # Manually map colors to include variable alpha for "cloud" effect
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(v_filtered)
        colors[:, 3] = v_filtered * 0.7 # Alpha based on intensity
        
        ax.scatter(p_filtered[:, 0], p_filtered[:, 1], p_filtered[:, 2], 
                  c=actual_vals_filtered, cmap=cmap_name, s=v_filtered*40, 
                  edgecolors='none', vmin=v_min, vmax=v_max)
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        # Set axes limits to [0, 1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # Show axes and labels
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=v_min, vmax=v_max))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.1)
        cb.set_label('Value', fontsize=12)

        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    add_cloud_subplot(ax1, u_test_sample, "True Solution", vmin=v_min_global, vmax=v_max_global)
    
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    add_cloud_subplot(ax2, u_pred_sample, "Predicted Solution", vmin=v_min_global, vmax=v_max_global)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    add_cloud_subplot(ax3, u_error, "Absolute Error")
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Volume visualization with correct axes and gradient colormap saved to {save_path}")

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
    results_dir = os.path.join(project_root, "experiments/results/poisson3d_lploss")
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
