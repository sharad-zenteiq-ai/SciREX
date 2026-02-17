"""
JAX implementation of 2D Darcy flow training for Wavelet Neural Operator (WNO).
Uses synthetic data generation via scirex.data.datasets.darcy.
Adapted from original PyTorch implementation: https://github.com/csccm-iitd/WNO
"""

import os
import argparse
import pickle
import json
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from flax.training import train_state

# Import existing WNO model from the repository
from scirex.operators.wno.models.wno2d import WNO2D
from scirex.operators.layers import Lifting, Projection
from scirex.data.datasets import darcy_zenodo
from scirex.training.normalizers import GaussianNormalizer
from scirex.data.datasets.darcy import generator as darcy_generator

# ==============================================================================
# CONFIGURATION / HYPERPARAMETERS
# ==============================================================================
class Config:
    # Dataset
    res = 128
    n_train = 1000
    n_test = 100
    data_dir = 'scirex/data/datasets/darcy_fno'
    
    # Model - Increased width for better representation
    width = 128
    depth = 4
    levels = 4 # 128 -> 8x8 coarse approximation
    wavelet = "db4"
    
    # Training
    batch_size = 32 # Leveraging GPU for larger batches
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-4
    seed = 42
    
    # Paths
    checkpoint_dir = 'experiments/checkpoints'
    checkpoint_filename = "darcy_wno_jax_best.pkl"

# ==============================================================================

# --- Loss Function ---

def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error loss."""
    return jnp.mean((pred - target) ** 2)

def relative_l2_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Relative L2 loss (standard for neural operators)."""
    # Flatten spatial dimensions: (Batch, N)
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    
    diff_norm = jnp.linalg.norm(pred_flat - target_flat, axis=-1)
    target_norm = jnp.linalg.norm(target_flat, axis=-1)
    return jnp.mean(diff_norm / (target_norm + 1e-7))

# --- Training State ---

class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, model, config, input_shape):
    """Initializes the training state with a scheduler (including warmup)."""
    variables = model.init(rng, jnp.ones(input_shape))
    params = variables['params']
    
    # Learning rate scheduler: Warmup + Cosine decay
    num_train_batches = config.n_train // config.batch_size
    total_steps = config.epochs * num_train_batches
    warmup_steps = 5 * num_train_batches # 5 epochs warmup
    
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=config.lr,
        decay_steps=total_steps - warmup_steps,
        alpha=0.1
    )
    
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, config.lr, warmup_steps),
            cosine_schedule
        ],
        boundaries=[warmup_steps]
    )
    
    # AdamW optimizer
    tx = optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

# --- JIT Compiled Steps ---

@jax.jit
def train_step(state: TrainState, a_batch: jnp.ndarray, u_batch: jnp.ndarray):
    """Performs a single training step using MSE loss."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, a_batch)
        loss = mse_loss(logits, u_batch)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (mse_loss_val, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Calculate Relative L2 for logging
    rel_l2 = relative_l2_loss(logits, u_batch)
    return state, mse_loss_val, rel_l2

@jax.jit
def eval_step(state: TrainState, a_batch: jnp.ndarray, u_batch: jnp.ndarray):
    """Performs an evaluation step reporting Relative L2 loss."""
    logits = state.apply_fn({'params': state.params}, a_batch)
    loss = relative_l2_loss(logits, u_batch)
    return loss

# --- Main Logic ---

def main():
    # 0. Check device
    print(f"JAX Devices: {jax.devices()}")

    # 1. Setup PRNG
    main_key = jax.random.PRNGKey(Config.seed)
    
    # 2. Load Data using Repo Dataloaders
    print(f"Loading Darcy dataset (res={Config.res})...")
    a_train_raw, u_train_raw, a_test_raw, u_test_raw = darcy_zenodo.load_darcy_numpy(
        root_dir=Config.data_dir,
        resolution=Config.res,
        n_train=Config.n_train,
        n_test=Config.n_test
    )
    
    # Coordinates
    x_grid = np.linspace(0, 1, Config.res)
    y_grid = np.linspace(0, 1, Config.res)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    grid = np.stack([X, Y], axis=-1)
    
    def append_grid(a):
        grid_batch = np.tile(grid[None, ...], (a.shape[0], 1, 1, 1))
        return np.concatenate([a, grid_batch], axis=-1)
    
    a_train_ext = append_grid(a_train_raw)
    a_test_ext = append_grid(a_test_raw)
    
    # 3. Normalization (Global statistics)
    print("Computing normalization statistics...")
    a_normalizer = GaussianNormalizer(a_train_ext)
    u_normalizer = GaussianNormalizer(u_train_raw)
    
    a_train_norm = a_normalizer.encode(jnp.asarray(a_train_ext))
    u_train_norm = u_normalizer.encode(jnp.asarray(u_train_raw))
    a_test_norm = a_normalizer.encode(jnp.asarray(a_test_ext))
    u_test_norm = u_normalizer.encode(jnp.asarray(u_test_raw))
    
    # 4. Initialize Data Generators (from repo)
    num_train_batches = Config.n_train // Config.batch_size
    num_test_batches = Config.n_test // Config.batch_size
    
    # 5. Initialize Model
    model = WNO2D(
        width=Config.width,
        depth=Config.depth,
        levels=Config.levels,
        wavelet=Config.wavelet,
        out_channels=1
    )
    
    input_shape = (Config.batch_size, Config.res, Config.res, 3)
    train_key, init_key = jax.random.split(main_key)
    state = create_train_state(init_key, model, Config, input_shape)
    
    # 6. Training Loop
    print(f"Starting training for {Config.epochs} epochs...")
    best_loss = float('inf')
    history_train = []
    history_test = []
    
    for epoch in range(1, Config.epochs + 1):
        # We manually batch here for simplicity with the loaded arrays
        indices = np.random.permutation(Config.n_train)
        train_losses = []
        
        for i in range(0, Config.n_train, Config.batch_size):
            idx = indices[i:i+Config.batch_size]
            if len(idx) < Config.batch_size: continue
            
            a_batch = a_train_norm[idx]
            u_batch = u_train_norm[idx]
            
            state, mse_v, rel_v = train_step(state, a_batch, u_batch)
            train_losses.append(rel_v)
            
            if (i // Config.batch_size) % 10 == 0:
                print(f"  Batch {i // Config.batch_size:3d}/{num_train_batches}: MSE {mse_v:.6f} | RelL2 {rel_v:.6f}")
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        test_losses = []
        for i in range(0, Config.n_test, Config.batch_size):
            idx = np.arange(i, min(i+Config.batch_size, Config.n_test))
            if len(idx) < Config.batch_size: continue
            
            a_batch = a_test_norm[idx]
            u_batch = u_test_norm[idx]
            
            loss = eval_step(state, a_batch, u_batch)
            test_losses.append(loss)
            
        avg_test_loss = np.mean(test_losses)
        history_train.append(float(avg_train_loss))
        history_test.append(float(avg_test_loss))
        
        print(f"Epoch {epoch:3d} | Train RelL2: {avg_train_loss:.6f} | Test RelL2: {avg_test_loss:.6f}")
        
        # Save Checkpoint
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            checkpoint = {
                'params': state.params,
                'a_norm': a_normalizer,
                'u_norm': u_normalizer,
                'config': {
                    'res': Config.res,
                    'width': Config.width,
                    'depth': Config.depth,
                    'levels': Config.levels,
                    'wavelet': Config.wavelet,
                    'seed': Config.seed
                }
            }
            save_path = Path(Config.checkpoint_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / Config.checkpoint_filename, "wb") as f:
                pickle.dump(checkpoint, f)

    # 7. Final Evaluation and Visualization
    print(f"✅ Training Complete. Best Test RelL2: {best_loss:.4f}")

    # Plot Loss Curve
    results_dir = Path("experiments/results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_train, label='Train RelL2')
    plt.plot(history_test, label='Test RelL2')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Loss')
    plt.title('WNO Training Progress (Darcy 2D)')
    plt.legend()
    plt.grid(True)
    plt.savefig(figures_dir / "loss_curve.png")
    print(f"📈 Loss curve saved to {figures_dir / 'loss_curve.png'}")

    # Save Metrics and Config to JSON
    metrics = {
        "best_test_rel_l2": float(best_loss),
        "final_train_rel_l2": history_train[-1],
        "final_test_rel_l2": history_test[-1],
        "config": {
            k: v for k, v in Config.__dict__.items() if not k.startswith("__")
        }
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📄 Metrics and config saved to {results_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
