import os
# Prevent JAX from pre-allocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.wno.models.wno2d import WNO2D
from scirex.operators.layers import Lifting, Projection
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import relative_l2_loss, mse
from scirex.data.datasets import darcy_zenodo
import optax
from scirex.training.normalizers import GaussianNormalizer
import pickle
from pathlib import Path

def main():
    # Model Config
    batch_size = 8 # Reduced to avoid OOM
    nx, ny, in_ch = 128, 128, 3 # (a, x, y)
    width = 64
    depth = 4
    levels = 6
    wavelet = "db4"
    out_ch = 1
    lr = 1e-3
    rng = jax.random.PRNGKey(42)

    # Pre-compute coordinate grids
    x_grid = np.linspace(0, 1, nx)
    y_grid = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    coords = np.stack([X, Y], axis=-1) # (nx, ny, 2)
    coords = jnp.array(coords)

    model = WNO2D(
        width=width, 
        depth=depth, 
        levels=levels, 
        wavelet=wavelet, 
        out_channels=out_ch
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
    state = create_train_state(rng, model, (batch_size, nx, ny, in_ch), tx=tx)

    # Dataset config - Using official Zenodo benchmark data (numpy format)
    data_dir = 'scirex/data/datasets/darcy'
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
        
        # Concatenate coordinates: [a, x, y]
        batch_coords = jnp.tile(coords[None, ...], (a_norm.shape[0], 1, 1, 1))
        x_input = jnp.concatenate([a_norm, batch_coords], axis=-1)
        
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
    
    batch_coords_test = jnp.tile(coords[None, ...], (a_norm_test.shape[0], 1, 1, 1))
    x_input_test = jnp.concatenate([a_norm_test, batch_coords_test], axis=-1)
    
    
    # Run evaluation first!
    eval_batch = {"x": x_input_test, "y": u_norm_test}
    final_metrics = eval_step(state, eval_batch, relative_l2_loss)

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
            "width": width,
            "depth": depth,
            "levels": levels,
            "wavelet": wavelet,
            "nx": nx,
            "ny": ny
        }
    }
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"✅ Model and normalizers saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
