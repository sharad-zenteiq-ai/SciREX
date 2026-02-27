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
JAX implementation of 2D Darcy flow training for Wavelet Neural Operator (WNO).
Faithfully follows original paper hyperparameters for rectangular domain.
Paper: Tripura & Chakraborty (2022)
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import pickle
import json
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
import matplotlib.pyplot as plt
from flax.training import train_state

from scirex.operators.models.wno2d import WNO2D
from scirex.data.datasets import darcy_zenodo
from scirex.training.normalizers import GaussianNormalizer
from scirex.data.datasets.darcy import generator as darcy_generator

# ==============================================================================
# CONFIGURATION / HYPERPARAMETERS (Paper Settings)
# ==============================================================================
class Config:
    # Dataset
    res = 128
    n_train = 1000
    n_test = 100
    data_dir = 'scirex/data/datasets/darcy_fno'
    
    # Model - Exact paper settings for rectangular Darcy flow
    hidden_channels = 64
    n_layers = 4
    levels = 4
    wavelet = "db4"
    in_channel = 1  # a(x,y)
    
    # Training
    batch_size = 20  
    epochs = 500    
    lr = 1e-3
    weight_decay = 1e-4
    gamma = 0.75      
    step_size = 50    
    seed = 42
    
    # Paths
    checkpoint_dir = 'experiments/checkpoints'
    checkpoint_filename = "darcy_wno_paper_best.pkl"

# ==============================================================================

# --- Loss Function ---

def relative_l2_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Relative L2 loss"""
    # Flatten spatial dimensions: (Batch, N)
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    
    diff_norm = jnp.linalg.norm(pred_flat - target_flat, axis=-1)
    target_norm = jnp.linalg.norm(target_flat, axis=-1)
    return jnp.mean(diff_norm / (target_norm + 1e-7))

# --- Training State ---

class TrainState(train_state.TrainState):
    # Store normalizers in state for easy decoding in JIT
    u_mean: jnp.ndarray
    u_std: jnp.ndarray

def create_train_state(rng, model, config, input_shape, u_normalizer):
    """Initializes the training state with a StepLR scheduler."""
    variables = model.init(rng, jnp.ones(input_shape))
    params = variables['params']
    
    num_train_batches = config.n_train // config.batch_size
    steps_per_epoch = num_train_batches
    
    schedule = optax.exponential_decay(
        init_value=config.lr,
        transition_steps=config.step_size * steps_per_epoch,
        decay_rate=config.gamma,
        staircase=True
    )
    
    # AdamW optimizer
    tx = optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        u_mean=jnp.array(u_normalizer.mean),
        u_std=jnp.array(u_normalizer.std)
    )

# --- JIT Compiled Steps ---

@jax.jit
def train_step(state: TrainState, a_batch: jnp.ndarray, u_batch_norm: jnp.ndarray):
    """Performs a single training step. Loss is calculated on DECODED scale."""
    def loss_fn(params):
        # 1. Prediction (normalized scale)
        pred_norm = state.apply_fn({'params': params}, a_batch)
        
        # 2. Decode to original scale (Paper does loss on physical values)
        pred = pred_norm * state.u_std + state.u_mean
        target = u_batch_norm * state.u_std + state.u_mean
        
        # 3. Calculate Relative L2 Loss
        loss = relative_l2_loss(pred, target)
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, pred), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss_val

@jax.jit
def eval_step(state: TrainState, a_batch: jnp.ndarray, u_batch_norm: jnp.ndarray):
    """Evaluation step reporting Relative L2 loss on decoded scale."""
    pred_norm = state.apply_fn({'params': state.params}, a_batch)
    
    pred = pred_norm * state.u_std + state.u_mean
    target = u_batch_norm * state.u_std + state.u_mean
    
    loss = relative_l2_loss(pred, target)
    return loss

# --- Main Logic ---

def main():
    print(f"JAX Devices: {jax.devices()}")

    main_key = jax.random.PRNGKey(Config.seed)
    
    # 2. Load Data
    print(f"Loading Darcy dataset (res={Config.res})...")
    a_train_raw, u_train_raw, a_test_raw, u_test_raw = darcy_zenodo.load_darcy_numpy(
        root_dir=Config.data_dir,
        resolution=Config.res,
        n_train=Config.n_train,
        n_test=Config.n_test
    )
    
    # 3. Normalization (Paper: Normalizes input and output)
    print("Computing normalization statistics...")
    a_normalizer = GaussianNormalizer(a_train_raw)
    u_normalizer = GaussianNormalizer(u_train_raw)
    
    a_train_norm = a_normalizer.encode(jnp.asarray(a_train_raw))
    u_train_norm = u_normalizer.encode(jnp.asarray(u_train_raw))
    a_test_norm = a_normalizer.encode(jnp.asarray(a_test_raw))
    u_test_norm = u_normalizer.encode(jnp.asarray(u_test_raw))
    
    # 4. Initialize Model (reference-aligned architecture)
    model = WNO2D(
        hidden_channels=Config.hidden_channels,
        n_layers=Config.n_layers,
        levels=Config.levels,
        wavelet=Config.wavelet,
        out_channels=1,
        use_grid=True
    )
    
    input_shape = (Config.batch_size, Config.res, Config.res, 1)  # Raw input: a(x,y) only
    train_key, init_key = jax.random.split(main_key)
    state = create_train_state(init_key, model, Config, input_shape, u_normalizer)
    
    # 6. Training Loop
    print(f"Starting training for {Config.epochs} epochs (Paper settings)...")
    
    initial_time = time.time()
    
    best_loss = float('inf')
    history_train = []
    history_test = []
    
    for epoch in range(1, Config.epochs + 1):
        indices = np.random.permutation(Config.n_train)
        train_losses = []
        
        for i in range(0, Config.n_train, Config.batch_size):
            idx = indices[i:i+Config.batch_size]
            if len(idx) < Config.batch_size: continue
            
            a_batch = a_train_norm[idx]
            u_batch = u_train_norm[idx]
            
            state, rel_l2_train = train_step(state, a_batch, u_batch)
            train_losses.append(rel_l2_train)
            
        avg_train_loss = np.mean(train_losses)
        
        # Validation every epoch
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
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Train RelL2: {avg_train_loss:.6f} | Test RelL2: {avg_test_loss:.6f}")
        
        # Save Best Checkpoint
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            checkpoint = {
                'params': state.params,
                'a_norm': a_normalizer,
                'u_norm': u_normalizer,
                'config': {
                    'res': Config.res,
                    'hidden_channels': Config.hidden_channels,
                    'n_layers': Config.n_layers,
                    'levels': Config.levels,
                    'wavelet': Config.wavelet
                }
            }
            save_path = Path(Config.checkpoint_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / Config.checkpoint_filename, "wb") as f:
                pickle.dump(checkpoint, f)

    print(f"Training Complete. Best Test RelL2: {best_loss:.4f}")

    # Plot and Save
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_train, label='Train RelL2')
    plt.plot(history_test, label='Test RelL2')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Loss')
    plt.title('WNO Paper Settings - Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / "paper_loss_curve.png")

    metrics = {
        "best_test_rel_l2": float(best_loss),
        "final_train_rel_l2": history_train[-1],
        "final_test_rel_l2": history_test[-1],
        "config": {k: v for k, v in Config.__dict__.items() if not k.startswith("__")}
    }
    with open(results_dir / "paper_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
