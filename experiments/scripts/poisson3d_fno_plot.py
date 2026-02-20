"""
Visualize FNO 3D Poisson Results.
Loads best checkpoint and plots slices of Ground Truth vs Prediction.
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

from scirex.operators.fno.models.fno3d import FNO3D
from scirex.data.datasets.poisson_3d import generator as poisson3d_generator
from experiments.configs.poisson.fno_config import FNO3DConfig

def main():
    config = FNO3DConfig()
    
    # 1. model
    model = FNO3D(
        width=config.width, 
        depth=config.depth, 
        modes_x=config.modes_x, 
        modes_y=config.modes_y, 
        modes_z=config.modes_z,
        out_channels=config.output_channels
    )
    
    # 2. Load Checkpoint
    ckpt_path = os.path.join(project_root, "experiments/checkpoints/poisson3d_fno_params.pkl")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    dummy_input = jnp.ones((1, config.nx, config.ny, config.nz, config.input_channels))
    variables = model.init(jax.random.PRNGKey(0), dummy_input)
    init_params = variables["params"]
    
    with open(ckpt_path, "rb") as f:
        ckpt_bytes = f.read()
    loaded_params = flax.serialization.from_bytes(init_params, ckpt_bytes)
    
    # 3. Data
    gen = poisson3d_generator(
        num_batches=1, batch_size=4, 
        nx=config.nx, ny=config.ny, nz=config.nz,
        include_mesh=config.include_mesh, rng_seed=12345
    )
    f_test, u_test = next(gen)
    
    # 4. Inference
    u_pred = model.apply({"params": loaded_params}, jnp.asarray(f_test))
    
    # 5. Plot Slices at z = nz // 2
    plot_dir = os.path.join(project_root, "experiments/results/poisson3d")
    os.makedirs(plot_dir, exist_ok=True)
    
    z_slice = config.nz // 2
    num_samples = 3
    fig, axes = plt.subplots(num_samples, 4, figsize=(18, 4 * num_samples))
    
    for i in range(num_samples):
        # f slice
        ax = axes[i, 0]
        im = ax.imshow(f_test[i, :, :, z_slice, 0], cmap="viridis")
        ax.set_title(f"Input f (z={z_slice})")
        fig.colorbar(im, ax=ax)
        
        # u Truth slice
        ax = axes[i, 1]
        im = ax.imshow(u_test[i, :, :, z_slice, 0], cmap="inferno")
        ax.set_title("Truth u")
        fig.colorbar(im, ax=ax)
        
        # u Pred slice
        ax = axes[i, 2]
        im = ax.imshow(u_pred[i, :, :, z_slice, 0], cmap="inferno")
        ax.set_title("Pred u_hat")
        fig.colorbar(im, ax=ax)
        
        # Error slice
        ax = axes[i, 3]
        error = np.abs(u_test[i, :, :, z_slice, 0] - u_pred[i, :, :, z_slice, 0])
        im = ax.imshow(error, cmap="magma")
        ax.set_title(f"Abs Error (Max: {error.max():.2e})")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, "poisson3d_fno_slices.png")
    plt.savefig(save_path, dpi=150)
    print(f"Slice plots saved to {save_path}")

    # 5b. 3D Volumetric Plot (Sample 0)
    # Showing 3D scatter to visualize the volume
    print("Generating 3D volumetric visualization...")
    fig_3d = plt.figure(figsize=(18, 6))
    
    # Grid coordinates for scatter
    x, y, z = np.meshgrid(np.arange(config.nx), np.arange(config.ny), np.arange(config.nz), indexing='ij')
    
    # We'll downsample slightly for clarity if needed, but 32 is okay
    # Just show one sample
    idx = 0
    
    def plot_3d_volume(ax, data, title, cmap="inferno"):
        # Explicitly calculate vmin/vmax to ensure good contrast
        vmin, vmax = np.min(data), np.max(data)
        
        sc = ax.scatter(x, y, z, c=data.flatten(), cmap=cmap, alpha=0.1, s=10, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Create a ScalarMappable for the colorbar with alpha=1.0
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig_3d.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Value', rotation=270, labelpad=15)
        
        return sc

    ax1 = fig_3d.add_subplot(1, 3, 1, projection='3d')
    plot_3d_volume(ax1, u_test[idx, ..., 0], "3D Truth")
    
    ax2 = fig_3d.add_subplot(1, 3, 2, projection='3d')
    plot_3d_volume(ax2, u_pred[idx, ..., 0], "3D Prediction")
    
    ax3 = fig_3d.add_subplot(1, 3, 3, projection='3d')
    err_3d = np.abs(u_test[idx, ..., 0] - u_pred[idx, ..., 0])
    plot_3d_volume(ax3, err_3d, "3D Abs Error", cmap="magma")
    
    plt.tight_layout()
    save_3d_path = os.path.join(plot_dir, "poisson3d_fno_volume.png")
    plt.savefig(save_3d_path, dpi=150)
    print(f"3D volumetric plots saved to {save_3d_path}")

    # 6. Loss Curves
    metrics_path = os.path.join(plot_dir, "fno3d_metrics.json")
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
        plt.title('3D Loss (MSE)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        # Plot Rel L2
        plt.subplot(1, 2, 2)
        if "train_rel_l2" in history:
            plt.semilogy(epochs_range, history["train_rel_l2"], label='Train Rel L2')
        plt.semilogy(epochs_range, history["test_rel_l2"], label='Test Rel L2', color='orange', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Relative L2 Error')
        plt.title('3D Accuracy (Rel L2)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "poisson3d_fno_losses.png"), dpi=150)
        print(f"Loss curves saved to: {os.path.join(plot_dir, 'poisson3d_fno_losses.png')}")
    else:
        print(f"Warning: Metrics file not found at {metrics_path}. Skipping loss plots.")

if __name__ == "__main__":
    main()
