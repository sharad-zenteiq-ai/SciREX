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
Visualize results from a trained JAX WNO model for the 2D Darcy problem.
"""

import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Import existing WNO model from the repository
from scirex.operators.models.wno2d import WNO2D
from scirex.training.train_state import create_train_state
from scirex.data.datasets import darcy_zenodo

def main():
    # 1. Load Checkpoint
    checkpoint_dir = Path("experiments/checkpoints")
    checkpoint_path = checkpoint_dir / "darcy_wno_jax_best.pkl"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}. Please run training first.")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    a_norm = checkpoint['a_norm']
    u_norm = checkpoint['u_norm']
    config = checkpoint['config']
    
    res = config['res']
    
    # 2. Load and Prepare Test Data
    print(f"Loading Darcy test data (res={res})...")
    _, _, a_test, u_test = darcy_zenodo.load_darcy_numpy(
        root_dir='scirex/data/datasets/darcy',
        resolution=res,
        n_train=1000,
        n_test=100
    )
    
    # Coordinates
    x_grid = np.linspace(0, 1, res)
    y_grid = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    grid = np.stack([X, Y], axis=-1)
    
    def append_grid(a):
        grid_batch = np.tile(grid[None, ...], (a.shape[0], 1, 1, 1))
        return np.concatenate([a, grid_batch], axis=-1)
    
    a_test_input = append_grid(a_test)
    
    # 3. Model Inference
    print("Running model inference...")
    model = WNO2D(
        hidden_channels=config['hidden_channels'],
        n_layers=config['n_layers'],
        levels=config['levels'],
        wavelet=config['wavelet'],
        out_channels=1,
        projection_hidden_dim=config.get('projection_hidden_dim', None)
    )
    
    # Pick a random sample from test set
    idx = np.random.randint(0, a_test.shape[0])
    # idx = 0 # Use first if preferred
    
    sample_a = a_test_input[idx:idx+1]
    sample_u = u_test[idx]
    
    # Encode input
    sample_a_norm = a_norm.encode(jnp.asarray(sample_a))
    
    # Predict
    preds_norm = model.apply({'params': params}, sample_a_norm)
    
    # Decode prediction
    u_pred_decoded = u_norm.decode(preds_norm)
    u_pred = np.array(u_pred_decoded[0, ..., 0])
    u_true = sample_u[..., 0]
    a_field = a_test[idx, ..., 0]
    
    # Calculate error
    error = np.abs(u_true - u_pred)
    rel_l2 = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    
    # 4. Plotting
    print("Generating plot...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 0. Permeability
    im0 = axes[0].imshow(a_field.T, origin='lower', cmap='viridis')
    axes[0].set_title(f"Input Permeability (a)\nResolution: {res}x{res}")
    plt.colorbar(im0, ax=axes[0])
    
    # 1. Ground Truth
    im1 = axes[1].imshow(u_true.T, origin='lower', cmap='jet')
    axes[1].set_title("Ground Truth (u)")
    plt.colorbar(im1, ax=axes[1])
    
    # 2. Prediction
    im2 = axes[2].imshow(u_pred.T, origin='lower', cmap='jet')
    axes[2].set_title(f"WNO Prediction\nRel. L2 Error: {rel_l2:.2%}")
    plt.colorbar(im2, ax=axes[2])
    
    # 3. Absolute Error
    im3 = axes[3].imshow(error.T, origin='lower', cmap='magma')
    axes[3].set_title(f"Absolute Error\nMax: {error.max():.4f}")
    plt.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    
    out_dir = Path("experiments/results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "darcy_wno_jax_inference.png"
    plt.savefig(out_path, dpi=150)
    print(f"✅ Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
