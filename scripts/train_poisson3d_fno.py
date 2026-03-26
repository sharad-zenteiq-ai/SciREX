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
3D Poisson Equation Solver Training Script using FNO and Relative L2 Loss.

This script learns the mapping f -> u for the 3D Poisson equation -∇²u = f.
It utilizes the FNO3D architecture and optimizes directly using the 
Relative L2 loss (LpLoss with p=2) on normalized scales.
"""
import os
import sys

# ── Force deterministic GPU operations for reproducibility ──
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# JAX Memory management: stop pre-allocating 90% of VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import linen as nn
import time
import matplotlib.pyplot as plt
import json

from scirex.operators.models.fno import FNO
from scirex.operators.training import create_train_state, TrainState
from scirex.operators.losses import lp_loss
from scirex.operators.data import random_poisson_3d_batch
from configs.poisson_fno_config import FNO3DConfig

def make_schedule(config: FNO3DConfig):
    """Create learning rate schedule with Linear Warmup and Cosine Decay."""
    spe = config.get_steps_per_epoch()
    total_steps = config.opt.n_epochs * spe
    
    # Warmup for ~5 epochs or 250 steps max
    warmup_steps = min(250, total_steps // 10)
    
    if config.opt.scheduler == "cosine":
        # Consistently decay over the specified cosine_decay_epochs
        cosine_decay_steps = config.opt.cosine_decay_epochs * spe - warmup_steps
        cosine_decay_steps = max(cosine_decay_steps, 1)
        
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=config.opt.learning_rate,
            decay_steps=cosine_decay_steps,
            alpha=0.0
        )
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, config.opt.learning_rate, warmup_steps),
                cosine_schedule
            ],
            boundaries=[warmup_steps]
        )
    elif config.opt.scheduler == "step":
        scales = {}
        decay_steps = config.opt.step_size * spe
        num_decays = config.opt.n_epochs // config.opt.step_size
        
        current_scale = 1.0
        for i in range(1, num_decays + 1):
            boundary = i * decay_steps
            current_scale *= config.opt.gamma
            scales[boundary] = current_scale
            
        schedule = optax.piecewise_constant_schedule(
            init_value=config.opt.learning_rate,
            boundaries_and_scales=scales
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.opt.scheduler}")
    
    return schedule

class UnitGaussianNormalizer:
    """Normalizer for 3D spatial fields."""
    def __init__(self, x, eps=1e-5):
        # x shape: (batch, nx, ny, nz, channels)
        # Normalize across batch and all spatial dimensions
        self.mean = jnp.mean(x, axis=(0, 1, 2, 3), keepdims=True)
        self.std = jnp.std(x, axis=(0, 1, 2, 3), keepdims=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

def main():
    # 1. Load Configuration
    config = FNO3DConfig()
    
    # Prng Key
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    # 2. Initialize Model
    print(f"Initializing FNO (hidden_channels={config.model.hidden_channels}, modes={config.model.n_modes})...")
    model = FNO(
        hidden_channels=config.model.hidden_channels, 
        n_layers=config.model.n_layers, 
        n_modes=config.model.n_modes, 
        out_channels=config.model.out_channels,
        lifting_channel_ratio=config.model.lifting_channel_ratio,
        projection_channel_ratio=config.model.projection_channel_ratio,
        use_grid=False, # Data already has grid (4 channels)
        use_norm=config.model.use_norm,
        fno_skip=config.model.fno_skip,
        channel_mlp_skip=config.model.channel_mlp_skip,
        use_channel_mlp=config.model.use_channel_mlp,
        padding=config.model.domain_padding,
        activation=nn.gelu
    )
    
    # 3. Initialize Optimizer & Scheduler
    schedule = make_schedule(config)
    steps_per_epoch = config.get_steps_per_epoch()
    total_steps = config.opt.n_epochs * steps_per_epoch
    
    # Create Train State
    # Data has 4 channels (1 source + 3 spatial coordinates)
    in_channels = 4
    nx, ny, nz = config.data.resolution
    input_shape = (config.data.batch_size, nx, ny, nz, in_channels)
    state = create_train_state(
        rng=init_rng, 
        model=model, 
        input_shape=input_shape, 
        learning_rate=schedule, 
        weight_decay=config.opt.weight_decay
    )
    
    # 4. Pre-generate Fixed Datasets
    n_train = config.data.n_train
    n_test = config.data.n_test
    
    print(f"Generating {n_train} training samples and {n_test} test samples (3D)...")
    
    # Generate Training Data
    f_train, u_train = random_poisson_3d_batch(
        batch_size=n_train, nx=nx, ny=ny, nz=nz, channels=1, 
        rng_seed=config.seed, include_mesh=True
    )
    # Generate Test Data
    f_test, u_test = random_poisson_3d_batch(
        batch_size=n_test, nx=nx, ny=ny, nz=nz, channels=1, 
        rng_seed=999, include_mesh=True
    )
    
    f_train = jnp.asarray(f_train)
    u_train = jnp.asarray(u_train)
    f_test = jnp.asarray(f_test)
    u_test = jnp.asarray(u_test)
    
    # 5. Normalize Data
    x_normalizer = UnitGaussianNormalizer(f_train)
    y_normalizer = UnitGaussianNormalizer(u_train)
    
    f_train_encoded = x_normalizer.encode(f_train)
    u_train_encoded = y_normalizer.encode(u_train)
    f_test_encoded = x_normalizer.encode(f_test)
    
    test_batch_encoded = {"x": f_test_encoded, "y": y_normalizer.encode(u_test)}
    
    print(f"Data shapes: f_train={f_train.shape}, u_train={u_train.shape}")
    
    # Paths
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoint_dir, config.model_name)

    os.makedirs(config.results_dir, exist_ok=True)

    best_rel_l2 = float("inf")
    history = {
        "train_rel_l2": [],
        "test_rel_l2": []
    }

    # 6. Training Step (Relative L2 on Encoded Scales)
    @jax.jit
    def train_step_lploss(state, batch):
        def loss_fn(params):
            pred_encoded = state.apply_fn({"params": params}, batch["x"])
            return lp_loss(pred_encoded, batch["y"])

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss}

    print(f"Starting training for {config.opt.n_epochs} epochs ({total_steps} steps)...")
    rng_key = jax.random.PRNGKey(config.seed + 1)
    
    _total_start_time = time.time()
    for epoch in range(config.opt.n_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Shuffle training data each epoch
        rng_key, shuffle_key = jax.random.split(rng_key)
        perm = jax.random.permutation(shuffle_key, n_train)
        f_shuffled = f_train_encoded[perm]
        u_shuffled = u_train_encoded[perm]
        
        for step in range(steps_per_epoch):
            start_idx = step * config.data.batch_size
            end_idx = start_idx + config.data.batch_size
            
            batch = {
                "x": f_shuffled[start_idx:end_idx],
                "y": u_shuffled[start_idx:end_idx]
            }
            
            state, metrics = train_step_lploss(state, batch)
            epoch_loss += float(metrics["loss"])
            
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = epoch_loss / steps_per_epoch
        
        # Evaluate on test set (decode for final metric)
        test_pred_encoded = state.apply_fn({"params": state.params}, test_batch_encoded["x"])
        test_pred_decoded = y_normalizer.decode(test_pred_encoded)
        
        v_test_l2 = float(lp_loss(test_pred_decoded, u_test))

        history["train_rel_l2"].append(avg_train_loss)
        history["test_rel_l2"].append(v_test_l2)
        
        # Save best checkpoint
        if v_test_l2 < best_rel_l2:
            best_rel_l2 = v_test_l2
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(state.params))
        
        current_lr = schedule(state.step)
        
        if epoch % 10 == 0 or epoch == config.opt.n_epochs - 1:
            print(f"Epoch {epoch:4d} | "
                  f"Train Rel L2: {avg_train_loss:.6e} | "
                  f"Test Rel L2: {v_test_l2:.6f} | "
                  f"Best Rel L2: {best_rel_l2:.6f} | "
                  f"LR: {float(current_lr):.2e} | "
                  f"Time: {epoch_time:.2f}s")
            
            with open(os.path.join(config.results_dir, "fno3d_metrics.json"), "w") as f:
                json.dump(history, f, indent=4)

    _total_end_time = time.time()
    total_time = _total_end_time - _total_start_time
    print("\nTraining Complete.")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Best Test Relative L2 Error: {best_rel_l2:.6f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # 7. Plot Loss Curves
    epochs_range = range(len(history["train_rel_l2"]))
    plt.figure(figsize=(8, 8))
    
    plt.semilogy(epochs_range, history["train_rel_l2"], label='Train Rel L2')
    plt.semilogy(epochs_range, history["test_rel_l2"], label='Test Rel L2', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error')
    plt.title('3D Poisson FNO: Relative L2 Error Convergence')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, "poisson3d_fno_losses.png"), dpi=150)
    print(f"Loss curves saved to: {os.path.join(config.results_dir, 'poisson3d_fno_losses.png')}")

if __name__ == "__main__":
    main()
