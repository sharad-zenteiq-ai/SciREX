"""
Train Fast Wavelet Neural Operator (FWNO) on Darcy Flow.
This script tests the new Lifting-based implementation with Rel L2 metric.
"""
import sys
import jax
import jax.numpy as jnp
from pathlib import Path
import optax
import numpy as np
import time
from flax.training import train_state

# Import the new Fast model
from scirex.operators.wno.models.fwno2d import FWNO2D
from scirex.data.datasets import darcy_zenodo
from scirex.training.normalizers import GaussianNormalizer

class Config:
    # Dataset (Standard Benchmark Settings)
    res = 128
    n_train = 1000
    n_test = 100
    data_dir = 'scirex/data/datasets/darcy_fno'
    
    # Model 
    width = 64
    depth = 4
    level = 4       # Match WNO paper
    in_channel = 3  # (a, x, y)
    
    # Training
    batch_size = 20
    epochs = 500
    lr = 1e-3
    weight_decay = 1e-4
    seed = 42

class TrainState(train_state.TrainState):
    u_mean: jnp.ndarray
    u_std: jnp.ndarray

def create_train_state(rng, model, input_shape, learning_rate, u_mean, u_std):
    """Creates initial `TrainState` with normalization stats."""
    params = model.init(rng, jnp.ones(input_shape))
    tx = optax.adamw(learning_rate, weight_decay=Config.weight_decay)
    return TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=tx,
        u_mean=u_mean,
        u_std=u_std
    )

def relative_l2_loss(pred, target):
    """Compute relative L2 loss."""
    # (B, H, W, C) -> (B, -1)
    diff_norms = jnp.linalg.norm((pred - target).reshape(pred.shape[0], -1), axis=1)
    target_norms = jnp.linalg.norm(target.reshape(target.shape[0], -1), axis=1)
    return jnp.mean(diff_norms / (target_norms + 1e-6))

@jax.jit
def train_step(state, batch):
    """Train step optimizing direct MSE but returning Rel L2."""
    def loss_fn(params):
        # 1. Predict (Normalized scale)
        pred_norm = state.apply_fn(params, batch['x'])
        
        # 2. Decode
        pred = pred_norm * state.u_std + state.u_mean
        target = batch['y_norm'] * state.u_std + state.u_mean
        
        # Optimize Rel L2 directly? WNO usually optimizes Rel L2.
        loss = relative_l2_loss(pred, target)
        return loss
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch):
    pred_norm = state.apply_fn(state.params, batch['x'])
    
    pred = pred_norm * state.u_std + state.u_mean
    target = batch['y_norm'] * state.u_std + state.u_mean
    
    return relative_l2_loss(pred, target)

def main():
    print("🚀 Starting FWNO Training (Fast Wavelet Neural Operator)...")
    
    # 1. Load Data
    try:
        a_train_raw, u_train_raw, a_test_raw, u_test_raw = darcy_zenodo.load_darcy_numpy(
            root_dir=Config.data_dir,
            resolution=Config.res,
            n_train=Config.n_train,
            n_test=Config.n_test
        )
    except FileNotFoundError:
        print(f"❌ Data not found. Please generate: python scripts/convert_darcy_zenodo.py {Config.res}")
        return

    # 2. Normalize
    a_normalizer = GaussianNormalizer(a_train_raw)
    u_normalizer = GaussianNormalizer(u_train_raw)
    
    a_train = a_normalizer.encode(jnp.array(a_train_raw))
    u_train_norm = u_normalizer.encode(jnp.array(u_train_raw))
    a_test = a_normalizer.encode(jnp.array(a_test_raw))
    u_test_norm = u_normalizer.encode(jnp.array(u_test_raw))
    
    u_mean = jnp.array(u_normalizer.mean)
    u_std = jnp.array(u_normalizer.std)
    
    # 3. Model
    rng = jax.random.PRNGKey(Config.seed)
    model = FWNO2D(
        width=Config.width,
        depth=Config.depth,
        channels=1,
        out_channels=1
    )
    
    # Init State
    input_shape = (1, Config.res, Config.res, 1)
    state = create_train_state(rng, model, input_shape, Config.lr, u_mean, u_std)
    print(f"✅ Model Initialized: FWNO (width={Config.width}, depth={Config.depth})")
    
    initial_time = time.time()
    
    # 4. Training Loop
    n_batches = Config.n_train // Config.batch_size
    
    for epoch in range(Config.epochs):
        epoch_loss = 0.0
        
        indices = np.random.permutation(Config.n_train)
        
        for i in range(n_batches):
            idx = indices[i*Config.batch_size : (i+1)*Config.batch_size]
            batch = {
                'x': a_train[idx],
                'y_norm': u_train_norm[idx] # Optimized on decoded but need norm input
            }
            state, loss = train_step(state, batch)
            epoch_loss += loss
            
        train_rel_l2 = epoch_loss / n_batches
        
        # Eval on full test set
        # Process in batches to avoid OOM if test set large
        test_rel_l2 = 0.0
        n_test_batches = Config.n_test // Config.batch_size
        if n_test_batches == 0: n_test_batches = 1
        
        for i in range(n_test_batches):
             start = i * Config.batch_size
             end = min((i+1) * Config.batch_size, Config.n_test)
             if start >= Config.n_test: break
             
             batch_test = {
                 'x': a_test[start:end],
                 'y_norm': u_test_norm[start:end]
             }
             test_rel_l2 += eval_step(state, batch_test)
             
        test_rel_l2 /= n_test_batches

        print(f"Epoch {epoch+1}/{Config.epochs} | Train RelL2: {train_rel_l2:.6f} | Test RelL2: {test_rel_l2:.6f}")
    
    final_time = time.time()
    print(f"Computational time: {final_time - initial_time}")

if __name__ == "__main__":
    main()
