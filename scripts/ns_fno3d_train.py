"""
Training script for Navier-Stokes 2D equation using FNO3D (JAX/Flax).

Dataset: nsforcing from neuraloperator Zenodo archive.
  - x: vorticity input (forcing), y: vorticity output (solution)
  - Each sample is a 2D field on a unit square grid.

Reference: https://github.com/neuraloperator/neuraloperator/blob/main/scripts/train_navier_stokes.py
  - The reference uses GINO3D for irregular data; since our data is on a regular
    unit square grid, we use FNO3D with a trivial z-dimension.
  - Training loss: LpLoss (relative L2), following the reference.
  - Evaluation: both MSE and relative L2.
  - Input normalization: disabled (reference default).
  - Output normalization: channel-wise Gaussian (reference default).
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
from pathlib import Path

# Force use of top-level scirex folder
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import torch
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from scirex.operators.models import FNO3D
from scirex.operators.training import GaussianNormalizer

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # Dataset Paths
    data_dir = '/media/HDD/mamta_backup/datasets/fno/navier_stokes'
    train_file = 'ns_train_64.pt'
    test_file = 'ns_test_64.pt'
    
    # Data (reference: encode both input and output)
    n_train = 1000  # Taking a subset of 10000 for faster training
    n_test = 200    # Taking a subset of 2000 for faster testing
    encode_input = True
    encode_output = True
    
    # Model Hyperparameters (reference FNO_Medium2d: modes=64, hidden=64)
    # Our data is 64x64, so modes = 32 (half of spatial dim)
    modes_x = 32
    modes_y = 32
    modes_z = 1   # z-dim is trivial (size 1) for 2D snapshots
    width = 64
    depth = 4
    
    # Training (reference: lr=3e-4, batch=8, epochs=600, StepLR step=100)
    batch_size = 8
    epochs = 100
    learning_rate = 3e-4
    weight_decay = 1e-4
    gamma = 0.5
    step_size = 100  # Decay LR every step_size epochs
    seed = 42
    
    # Checkpoint
    checkpoint_dir = 'experiments/checkpoints'
    model_name = "ns_fno3d_jax.pkl"

# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

def relative_l2_loss(pred, target):
    """Relative L2 loss (LpLoss with d=2, p=2)."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    diff_norm = jnp.linalg.norm(pred_flat - target_flat, axis=-1)
    target_norm = jnp.linalg.norm(target_flat, axis=-1)
    return jnp.mean(diff_norm / (target_norm + 1e-7))

def mse_loss(pred, target):
    """Mean Squared Error loss."""
    return jnp.mean((pred - target) ** 2)

def h1_loss(pred, target):
    """Relative H1 Sobolev loss (matches neuraloperator H1Loss(d=2).__call__).
    
    Computes: sum_batch[ sqrt(diff) / (sqrt(ynorm) + eps) ]
    where diff = ||pred-target||^2 + ||d(pred-target)/dx||^2 + ||d(pred-target)/dy||^2
    and ynorm = ||target||^2 + ||dtarget/dx||^2 + ||dtarget/dy||^2
    
    Uses central finite differences (periodic BC) for spatial derivatives.
    Reduction: sum over batch (reference default).
    """
    eps = 1e-8
    
    # Flatten spatial dims for the L2 part: (batch, nx, ny, nz, ch) -> (batch, -1)
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    
    # L2 terms (per sample)
    diff_l2 = jnp.sum((pred_flat - target_flat) ** 2, axis=-1)
    ynorm_l2 = jnp.sum(target_flat ** 2, axis=-1)
    
    # Spatial derivatives via finite differences along x (dim 1) and y (dim 2)
    # Using periodic central differences: df/dx[i] = (f[i+1] - f[i-1]) / 2h
    # For simplicity with h=1/N (unit domain), scale is absorbed into the norm
    dx_pred = jnp.roll(pred, -1, axis=1) - jnp.roll(pred, 1, axis=1)
    dx_target = jnp.roll(target, -1, axis=1) - jnp.roll(target, 1, axis=1)
    dy_pred = jnp.roll(pred, -1, axis=2) - jnp.roll(pred, 1, axis=2)
    dy_target = jnp.roll(target, -1, axis=2) - jnp.roll(target, 1, axis=2)
    
    # Flatten gradient terms
    dx_diff_flat = (dx_pred - dx_target).reshape(pred.shape[0], -1)
    dx_target_flat = dx_target.reshape(pred.shape[0], -1)
    dy_diff_flat = (dy_pred - dy_target).reshape(pred.shape[0], -1)
    dy_target_flat = dy_target.reshape(pred.shape[0], -1)
    
    # Gradient norm terms (per sample)
    diff_dx = jnp.sum(dx_diff_flat ** 2, axis=-1)
    ynorm_dx = jnp.sum(dx_target_flat ** 2, axis=-1)
    diff_dy = jnp.sum(dy_diff_flat ** 2, axis=-1)
    ynorm_dy = jnp.sum(dy_target_flat ** 2, axis=-1)
    
    # Total H1 norm: value + gradients
    diff_total = diff_l2 + diff_dx + diff_dy
    ynorm_total = ynorm_l2 + ynorm_dx + ynorm_dy
    
    # Relative norm per sample, then sum (reference default reduction='sum')
    h1_per_sample = jnp.sqrt(diff_total) / (jnp.sqrt(ynorm_total) + eps)
    return jnp.sum(h1_per_sample)

# ==============================================================================
# TRAINING UTILS
# ==============================================================================

class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, model, config, input_shape):
    variables = model.init(rng, jnp.ones(input_shape))
    params = variables['params']
    
    # StepLR schedule (reference uses StepLR or CosineAnnealing)
    num_train_batches = config.n_train // config.batch_size
    lr_schedule = optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=config.step_size * num_train_batches,
        decay_rate=config.gamma,
        staircase=True
    )
    
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch_x, batch_y):
    """Train on H1 loss (reference default), track RelL2 and MSE."""
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, batch_x)
        loss = h1_loss(pred, batch_y)
        rel_l2 = relative_l2_loss(pred, batch_y)
        mse = mse_loss(pred, batch_y)
        return loss, (rel_l2, mse)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (rel_l2, mse)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, rel_l2, mse

@jax.jit
def eval_step(state, batch_x, batch_y):
    pred = state.apply_fn({'params': state.params}, batch_x)
    rel_l2 = relative_l2_loss(pred, batch_y)
    mse = mse_loss(pred, batch_y)
    return rel_l2, mse

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_ns_data(config):
    """
    Load Navier-Stokes .pt files (dict with 'x' and 'y' keys).
    Reference: encode_input=False, encode_output=True (channel-wise Gaussian).
    """
    print(f"Loading data from {config.data_dir}/{config.train_file}...")
    
    train_dict = torch.load(os.path.join(config.data_dir, config.train_file), map_location='cpu')
    test_dict = torch.load(os.path.join(config.data_dir, config.test_file), map_location='cpu')
    
    # Raw shapes: (N, H, W) -> take n_train/n_test samples
    train_x = train_dict['x'][:config.n_train].numpy()  # (1000, 64, 64)
    train_y = train_dict['y'][:config.n_train].numpy()
    test_x = test_dict['x'][:config.n_test].numpy()      # (200, 64, 64)
    test_y = test_dict['y'][:config.n_test].numpy()
    
    # Add z-dim (size 1) and channel dim for FNO3D: (N, H, W) -> (N, H, W, 1, 1)
    train_x = train_x[:, :, :, np.newaxis, np.newaxis]
    train_y = train_y[:, :, :, np.newaxis, np.newaxis]
    test_x = test_x[:, :, :, np.newaxis, np.newaxis]
    test_y = test_y[:, :, :, np.newaxis, np.newaxis]
    
    print(f"  Train: x={train_x.shape}, y={train_y.shape}")
    print(f"  Test:  x={test_x.shape}, y={test_y.shape}")
    print(f"  X range: [{train_x.min():.4f}, {train_x.max():.4f}]")
    print(f"  Y range: [{train_y.min():.4f}, {train_y.max():.4f}]")
    
    return train_x, train_y, test_x, test_y

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    config = Config()
    print(f"JAX devices: {jax.devices()}")
    
    # 1. Load Data
    train_x, train_y, test_x, test_y = load_ns_data(config)
    
    # 2. Normalize (reference: only output, channel-wise Gaussian)
    if config.encode_output:
        y_normalizer = GaussianNormalizer(train_y)
        train_y = y_normalizer.encode(train_y)
        test_y = y_normalizer.encode(test_y)
        print(f"  Output normalized (channel-wise Gaussian)")
    
    if config.encode_input:
        x_normalizer = GaussianNormalizer(train_x)
        train_x = x_normalizer.encode(train_x)
        test_x = x_normalizer.encode(test_x)
        print(f"  Input normalized (channel-wise Gaussian)")
    
    # Convert to jnp
    train_x, train_y = jnp.array(train_x), jnp.array(train_y)
    test_x, test_y = jnp.array(test_x), jnp.array(test_y)
    
    # 3. Model
    model = FNO3D(
        hidden_channels=config.width,
        n_layers=config.depth,
        n_modes=(config.modes_x, config.modes_y, config.modes_z),
        out_channels=1
    )
    
    key = jax.random.PRNGKey(config.seed)
    input_shape = train_x[:1].shape
    state = create_train_state(key, model, config, input_shape)
    
    # Count params
    n_params = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"  Model parameters: {n_params:,}")
    
    # 4. Training Loop
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"{'Epoch':>6} | {'Train RelL2':>12} {'Train MSE':>12} | {'Test RelL2':>12} | {'Time':>8}")
    print("-" * 75)
    
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]
    best_test_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        # Shuffle
        idx = np.random.permutation(num_train)
        train_err = 0.0
        n_train_samples = 0
        
        start_time = time.time()
        for i in range(0, num_train, config.batch_size):
            batch_idx = idx[i:i+config.batch_size]
            if len(batch_idx) < config.batch_size:
                continue
            
            bx, by = train_x[batch_idx], train_y[batch_idx]
            state, rel_l2, mse = train_step(state, bx, by)
            train_err += float(rel_l2)
            n_train_samples += len(batch_idx)
        
        epoch_time = time.time() - start_time
        num_batches = max(n_train_samples // config.batch_size, 1)
        avg_loss = train_err / max(n_train_samples, 1)
        train_err_avg = train_err / num_batches
        
        # Eval: compute both h1 and l2 on test (reference evaluates both)
        test_h1_total = 0.0
        test_l2_total = 0.0
        n_test_samples = 0
        for i in range(0, num_test, config.batch_size):
            batch_idx = np.arange(i, min(i+config.batch_size, num_test))
            if len(batch_idx) == 0:
                continue
            bx, by = test_x[batch_idx], test_y[batch_idx]
            pred = state.apply_fn({'params': state.params}, bx)
            batch_h1 = float(h1_loss(pred, by))
            batch_l2 = float(relative_l2_loss(pred, by))
            test_h1_total += batch_h1
            test_l2_total += batch_l2
            n_test_samples += len(batch_idx)
            
        test_h1_avg = test_h1_total / max(n_test_samples, 1)
        test_l2_avg = test_l2_total / max(n_test_samples, 1)
        
        if test_l2_avg < best_test_loss:
            best_test_loss = test_l2_avg
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch}] time={epoch_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err_avg:.4f}")
            print(f"Eval: 64_h1={test_h1_avg:.4f}, 64_l2={test_l2_avg:.4f}")

    print(f"\nTraining finished. Best Test L2: {best_test_loss:.6f}")

    # 5. Save checkpoint
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"Model ready in memory.")

if __name__ == "__main__":
    main()
