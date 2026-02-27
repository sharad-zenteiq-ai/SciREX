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

import os
# Prevent JAX from pre-allocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path

# Force use of top-level scirex folder
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.models.wno import WNO2D
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import lp_loss, mse
from scirex.data.datasets import darcy_zenodo
import optax
from scirex.training.normalizers import GaussianNormalizer
import pickle
from pathlib import Path

def main():
    # Model Config (matching reference WNO paper settings)
    batch_size = 8 # Reduced to avoid OOM
    nx, ny, in_channels = 128, 128, 1 # Raw input (a only)
    hidden_channels = 64
    n_layers = 4
    levels = 6
    wavelet = "db4"
    out_channels = 1
    lr = 1e-3
    rng = jax.random.PRNGKey(42)

    model = WNO2D(
        hidden_channels=hidden_channels, 
        n_layers=n_layers, 
        levels=levels, 
        wavelet=wavelet, 
        out_channels=out_channels,
        use_grid=True
    )
    # Scheduler: Warmup + Cosine Decay
    num_batches = 5000
    warmup_steps = 500
    
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=num_batches,
        end_value=1e-6
    )
    
    tx = optax.adam(learning_rate=scheduler)
    state = create_train_state(rng, model, (batch_size, nx, ny, in_channels), tx=tx)

    # Dataset config - Using official Zenodo benchmark data (numpy format)
    data_dir = 'scirex/data/datasets/darcy_fno'
    gen = darcy_zenodo.generator_from_numpy(
        root_dir=data_dir,
        resolution=nx,
        n_train=1000,
        batch_size=batch_size,
        num_batches=num_batches,
        shuffle=True
    )

    # 1. Compute normalization stats from a few batches
    print("Computing normalization stats...")
    warmup_gen = darcy_zenodo.generator_from_numpy(
        root_dir=data_dir, resolution=nx, n_train=1000, batch_size=batch_size, num_batches=20, shuffle=True
    )
    a_samples, u_samples = [], []
    for a_s, u_s in warmup_gen:
        a_samples.append(a_s)
        u_samples.append(u_s)
    
    a_normalizer = GaussianNormalizer(jnp.concatenate(a_samples, axis=0))
    u_normalizer = GaussianNormalizer(jnp.concatenate(u_samples, axis=0))

    print(f"Starting training (WNO 2D Darcy + Normalization)...")
    for step, (a_np, u_np) in enumerate(gen):
        # Normalize
        a_norm = a_normalizer.encode(jnp.asarray(a_np))
        u_norm = u_normalizer.encode(jnp.asarray(u_np))
        
        # WNO2D now appends grid coordinates internally
        x_input = a_norm
        
        batch = {"x": x_input, "y": u_norm}
        state, metrics = train_step(state, batch, mse)
        
        if step % 10 == 0:
            print(f"step {step:5d}, loss: {float(metrics['loss']):.6e}")
            
        if step == num_batches - 1 or (step > 0 and step % 50 == 0):
            eval_metrics = eval_step(state, batch, mse)
            print(f"  eval loss at step {step}: {float(eval_metrics['loss']):.6e}")

    # Final validation on fresh data
    print("Final evaluation...")
    # Load test data from numpy files
    _, _, a_test_all, u_test_all = darcy_zenodo.load_darcy_numpy(
        root_dir=data_dir,
        resolution=nx,
        n_train=1000,
        n_test=100
    )
    # Use first batch_size samples from test set
    a_test = a_test_all[:batch_size]
    u_test = u_test_all[:batch_size]
    
    a_norm_test = a_normalizer.encode(jnp.asarray(a_test))
    u_norm_test = u_normalizer.encode(jnp.asarray(u_test))
    
    x_input_test = a_norm_test
    
    # Run evaluation first!
    eval_batch = {"x": x_input_test, "y": u_norm_test}
    final_metrics = eval_step(state, eval_batch, lp_loss)

    # --- Detailed Metrics Calculation ---
    # 1. Decode predictions and ground truth to original scale
    preds_norm = final_metrics["preds"]  # (B, 128, 128, 1)
    targets_norm = eval_batch["y"]       # (B, 128, 128, 1)
    
    preds_decoded = u_normalizer.decode(preds_norm)
    targets_decoded = u_normalizer.decode(targets_norm)
    
    # 2. Flatten for metrics
    pred_flat = preds_decoded.reshape(-1)
    true_flat = targets_decoded.reshape(-1)
    
    # 3. Calculate metrics
    mse_val = np.mean((pred_flat - true_flat) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(pred_flat - true_flat))
    
    # Relative L2 Error (Standard Metric for PDEs)
    # Norm of error / Norm of truth
    # Calculated per sample then averaged
    error_norms = np.linalg.norm(preds_decoded.reshape(batch_size, -1) - targets_decoded.reshape(batch_size, -1), axis=1)
    true_norms = np.linalg.norm(targets_decoded.reshape(batch_size, -1), axis=1)
    rel_l2 = np.mean(error_norms / true_norms)
    
    # R2 Score
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    print("\n" + "="*40)
    print("FINAL RESULTS (128x128 Darcy)")
    print("="*40)
    print(f"MSE:          {mse_val:.6e}")
    print(f"RMSE:         {rmse_val:.6e}")
    print(f"MAE:          {mae_val:.6e}")
    print(f"Relative L2:  {rel_l2:.4%}  (Benchmark Metric)")
    print(f"R2 Score:     {r2_score:.4f}")
    print("="*40 + "\n")

    # --- Save Model and Normalizers ---
    checkpoint_dir = Path("experiments/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "darcy_wno_128.pkl"
    
    save_dict = {
        "params": state.params,
        "a_normalizer": a_normalizer,
        "u_normalizer": u_normalizer,
        "config": {
            "hidden_channels": hidden_channels,
            "n_layers": n_layers,
            "levels": levels,
            "wavelet": wavelet,
            "nx": nx,
            "ny": ny
        }
    }
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Model and normalizers saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
