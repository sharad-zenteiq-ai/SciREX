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

"""Load trained WNO2D model and plot results for the 2D Darcy problem."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from scirex.operators.models.wno2d import WNO2D
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import eval_step
from scirex.losses.data_losses import mse
from scirex.data.datasets import darcy_zenodo

def main():
    # 1. Load Checkpoint
    checkpoint_path = Path("experiments/checkpoints/darcy_wno_128.pkl")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}. Please run training first.")
        return

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    
    config = checkpoint["config"]
    params = checkpoint["params"]
    a_normalizer = checkpoint["a_normalizer"]
    u_normalizer = checkpoint["u_normalizer"]
    
    nx, ny = config["nx"], config["ny"]
    hidden_channels = config["hidden_channels"]
    n_layers = config["n_layers"]
    levels = config["levels"]
    wavelet = config["wavelet"]
    
    print(f"✅ Loaded model: {wavelet} hidden_channels={hidden_channels} n_layers={n_layers} levels={levels} res={nx}x{ny}")

    # 2. Initialize Model and State
    model = WNO2D(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        levels=levels, 
        wavelet=wavelet, 
        out_channels=1
    )
    
    # Dummy init to create state structure
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, (1, nx, ny, 3), learning_rate=1e-3)
    # Replace params with loaded ones
    state = state.replace(params=params)

    # 3. Pre-compute coordinate grids
    x_grid = np.linspace(0, 1, nx)
    y_grid = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    coords = jnp.array(np.stack([X, Y], axis=-1)) # (nx, ny, 2)

    # 4. Load Test Data
    data_dir = 'scirex/data/datasets/darcy'
    _, _, a_test_all, u_test_all = darcy_zenodo.load_darcy_numpy(
        root_dir=data_dir,
        resolution=nx,
        n_train=1000,
        n_test=100
    )
    
    # Pick a random sample from test set
    idx = 0 # Can change to np.random.randint(len(a_test_all))
    a_test = a_test_all[idx:idx+1]
    u_test = u_test_all[idx:idx+1]
    
    # 5. Inference
    a_norm_test = a_normalizer.encode(jnp.asarray(a_test))
    u_norm_test = u_normalizer.encode(jnp.asarray(u_test))
    
    batch_coords_test = jnp.tile(coords[None, ...], (1, 1, 1, 1))
    x_input_test = jnp.concatenate([a_norm_test, batch_coords_test], axis=-1)
    
    eval_batch = {"x": x_input_test, "y": u_norm_test}
    out = eval_step(state, eval_batch, mse)
    
    # Decode results
    u_pred_norm = out["preds"][0] # (128, 128, 1)
    u_pred_decoded = u_normalizer.decode(u_pred_norm)
    u_pred = np.array(u_pred_decoded[..., 0])
    
    u_true = u_test[0, ..., 0]
    a_field = a_test[0, ..., 0]

    # 6. Calculate Detailed Metrics
    mse_val = np.mean((u_pred - u_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(u_pred - u_true))
    # Relative L2: ||u_pred - u_true||_2 / ||u_true||_2
    rel_l2 = np.linalg.norm(u_pred.flatten() - u_true.flatten()) / np.linalg.norm(u_true.flatten())
    # R2 Score
    ss_res = np.sum((u_true - u_pred) ** 2)
    ss_tot = np.sum((u_true - np.mean(u_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    print("\n" + "="*40)
    print("INFERENCE RESULTS")
    print("="*40)
    print(f"MSE:          {mse_val:.6e}")
    print(f"RMSE:         {rmse_val:.6e}")
    print(f"MAE:          {mae_val:.6e}")
    print(f"Relative L2:  {rel_l2:.4%}")
    print(f"R2 Score:     {r2_score:.4f}")
    print("="*40 + "\n")

    # 7. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Permeability
    im0 = axes[0].imshow(a_field.T, origin='lower', cmap='viridis')
    axes[0].set_title(f'Permeability (a)\nInput')
    plt.colorbar(im0, ax=axes[0])
    
    # True Pressure
    im1 = axes[1].imshow(u_true.T, origin='lower', cmap='jet')
    axes[1].set_title(f'True Pressure (u)\nTarget')
    plt.colorbar(im1, ax=axes[1])
    
    # Predicted Pressure
    im2 = axes[2].imshow(u_pred.T, origin='lower', cmap='jet')
    axes[2].set_title(f'WNO Prediction\nRel. L2: {rel_l2:.2%}')
    plt.colorbar(im2, ax=axes[2])
    
    # Error Map
    error_map = np.abs(u_true - u_pred)
    im3 = axes[3].imshow(error_map.T, origin='lower', cmap='magma')
    axes[3].set_title(f'Absolute Error\nMAE: {mae_val:.4f}')
    plt.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    out_dir = Path("experiments/results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"darcy_wno_inference_128.png"
    plt.savefig(fig_path, dpi=150)
    print(f"📊 Plate saved to {fig_path}")

if __name__ == "__main__":
    main()
