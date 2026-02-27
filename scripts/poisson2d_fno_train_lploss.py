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

"""Train a Flax FNO to learn the Poisson solution operator (f -> u)."""
import os
import sys

# ── Force deterministic GPU operations for reproducibility ──
# Must be set BEFORE importing JAX so XLA picks it up at init time.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import linen as nn
import time
import matplotlib.pyplot as plt
import json

from scirex.operators.models.fno import FNO2D
from scirex.training.train_state import create_train_state, TrainState
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse, lp_loss
from scirex.data.datasets.poisson import generator as poisson_generator
from configs.poisson_fno_config import FNO2DConfig


def make_schedule(config: FNO2DConfig):
    """Create learning rate schedule with Linear Warmup and Cosine Decay.
    
    The cosine decay is over `cosine_decay_epochs` (default 100). Any epochs
    beyond that train at near-zero LR (fine-tuning). This means increasing
    `epochs` never degrades performance — it only adds bonus polishing time.
    """
    spe = getattr(config, "steps_per_epoch_actual", config.steps_per_epoch)
    total_steps = config.epochs * spe
    
    # Fixed warmup: ~10 epochs worth of steps
    warmup_steps = min(310, total_steps // 10)
    
    if config.scheduler_type == "cosine":
        # Cosine decay length is FIXED to cosine_decay_epochs, NOT total epochs.
        cosine_decay_steps = config.cosine_decay_epochs * spe - warmup_steps
        cosine_decay_steps = max(cosine_decay_steps, 1)  # safety
        
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=cosine_decay_steps,
            alpha=0.0
        )
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
                cosine_schedule
            ],
            boundaries=[warmup_steps]
        )
    elif config.scheduler_type == "step":
        scales = {}
        decay_steps = config.scheduler_step_size * spe
        num_decays = config.epochs // config.scheduler_step_size
        
        current_scale = 1.0
        for i in range(1, num_decays + 1):
            boundary = i * decay_steps
            current_scale *= config.scheduler_gamma
            scales[boundary] = current_scale
            
        schedule = optax.piecewise_constant_schedule(
            init_value=config.learning_rate,
            boundaries_and_scales=scales
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {config.scheduler_type}")
    
    return schedule

class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        self.mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
        self.std = jnp.std(x, axis=(0, 1, 2), keepdims=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

def main():
    # 1. Load Configuration
    config = FNO2DConfig()
    
    # Prng Key
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    # 2. Initialize Model
    print(f"Initializing FNO2D (hidden_channels={config.hidden_channels}, modes={config.n_modes})...")
    model = FNO2D(
        hidden_channels=config.hidden_channels, 
        n_layers=config.n_layers, 
        n_modes=config.n_modes, 
        out_channels=config.out_channels,
        lifting_channel_ratio=config.lifting_channel_ratio,
        projection_channel_ratio=config.projection_channel_ratio,
        use_grid=config.use_grid,
        fno_skip=config.fno_skip,
        channel_mlp_skip=config.channel_mlp_skip,
        use_channel_mlp=config.use_channel_mlp,
        padding=config.domain_padding,
        activation=nn.gelu
    )
    
    # Create checkpoint directory and results directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 3. Initialize Optimizer & Scheduler
    # Override steps_per_epoch based on actual n_train if provided
    n_train = getattr(config, "n_train", config.batch_size * config.steps_per_epoch)
    steps_per_epoch = n_train // config.batch_size
    setattr(config, "steps_per_epoch_actual", steps_per_epoch)
    
    schedule = make_schedule(config)
    # Total steps verification
    total_steps = config.epochs * steps_per_epoch
    
    # Create Train State
    # Shape: (batch, nx, ny, in_channels)
    nx, ny = config.resolution
    input_shape = (config.batch_size, nx, ny, config.in_channels)
    state = create_train_state(
        rng=init_rng, 
        model=model, 
        input_shape=input_shape, 
        learning_rate=schedule, 
        weight_decay=config.weight_decay
    )
    
    # 4. Pre-generate FIXED training and test datasets
    # The original FNO paper uses fixed datasets (not infinite random streams).
    # A fixed dataset allows the model to learn fine-grained patterns.
    # Prioritize config.n_train if it exists, otherwise fallback to batch_size * steps_per_epoch.
    n_test = config.n_test  # should be 200
    
    print(f"Generating {n_train} training samples and {n_test} test samples...")
    from scirex.data.datasets.poisson import random_poisson_batch
    
    # Generate all training data at once
    f_train, u_train = random_poisson_batch(
        batch_size=n_train, nx=nx, ny=ny, channels=1, rng_seed=config.seed
    )
    # Generate test data
    f_test, u_test = random_poisson_batch(
        batch_size=n_test, nx=nx, ny=ny, channels=1, rng_seed=999
    )
    
    f_train = jnp.asarray(f_train)
    u_train = jnp.asarray(u_train)
    f_test = jnp.asarray(f_test)
    u_test = jnp.asarray(u_test)
    
    # 5. Normalize Data globally (standard in neural operator)
    x_normalizer = UnitGaussianNormalizer(f_train)
    y_normalizer = UnitGaussianNormalizer(u_train)
    
    f_train = x_normalizer.encode(f_train)
    u_train = y_normalizer.encode(u_train)
    f_test_encoded = x_normalizer.encode(f_test)
    # Note: For evaluation, evaluate on un-normalized targets!
    # The neuraloperator predicts encoded 'y', then decodes back.
    # We will test using un-normalized target.
    test_batch_encoded = {"x": f_test_encoded, "y": y_normalizer.encode(u_test)}
    test_batch_unencoded = {"x": f_test_encoded, "y": u_test}
    print(f"Data shapes: f_train={f_train.shape}, u_train={u_train.shape}")
    print(f"Data stats: f std={float(f_train[...,0].std()):.4f}, u std={float(u_train.std()):.4f}")
    
    # Create checkpoint directory and results directory
    ckpt_dir = os.path.join(project_root, "experiments/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "poisson_fno_params.pkl")

    results_dir = os.path.join(project_root, "experiments/results/experiment")
    os.makedirs(results_dir, exist_ok=True)

    best_rel_l2 = float("inf")
    
    # Track metrics for plotting
    history = {
        "train_rel_l2": [],
        "test_rel_l2": []
    }

    # 5. Training Step (Optimized for Gradient Stability)
    @jax.jit
    def train_step_lploss(state, batch):
        def loss_fn(params):
            # Optimizing for Relative L2 on the encoded scale 
            # (where gradients are better conditioned O(1))
            pred_encoded = state.apply_fn({"params": params}, batch["x"])
            return lp_loss(pred_encoded, batch["y"])

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss}

    print(f"Starting training for {config.epochs} epochs ({total_steps} steps)...")
    rng_key = jax.random.PRNGKey(config.seed + 1)
    
    # Final check of data shapes
    print(f"Data shapes: f_train={f_train.shape}, u_train={u_train.shape}")
    print(f"Test shapes: f_test={test_batch_encoded['x'].shape}, u_test={u_test.shape}")

    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Shuffle training data each epoch
        rng_key, shuffle_key = jax.random.split(rng_key)
        perm = jax.random.permutation(shuffle_key, n_train)
        f_shuffled = f_train[perm]
        u_shuffled = u_train[perm]
        
        for step in range(steps_per_epoch):
            start_idx = step * config.batch_size
            end_idx = start_idx + config.batch_size
            
            batch = {
                "x": f_shuffled[start_idx:end_idx],
                "y": u_shuffled[start_idx:end_idx]
            }
            
            state, metrics = train_step_lploss(state, batch)
            epoch_loss += float(metrics["loss"])
            
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = epoch_loss / steps_per_epoch
        
        # Evaluate on test set
        # Forward pass output is "encoded" y.
        test_pred_encoded = state.apply_fn({"params": state.params}, test_batch_encoded["x"])
        test_pred_decoded = y_normalizer.decode(test_pred_encoded)
        
        # Test metric on original scale
        v_test_l2 = float(lp_loss(test_pred_decoded, u_test))

        # Update history
        history["train_rel_l2"].append(avg_train_loss)
        history["test_rel_l2"].append(v_test_l2)
        
        test_rel_l2 = v_test_l2 # for checkpointing
        
        # Save best checkpoint
        if test_rel_l2 < best_rel_l2:
            best_rel_l2 = test_rel_l2
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(state.params))
        
        # Scheduler info
        current_lr = schedule(state.step)
        
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:4d} | "
                  f"Train Rel L2: {avg_train_loss:.6e} | "
                  f"Test Rel L2: {v_test_l2:.6f} | "
                  f"Best Rel L2: {best_rel_l2:.6f} | "
                  f"LR: {float(current_lr):.2e} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save history periodically to avoid loss on interruption
            with open(os.path.join(results_dir, "fno2d_metrics.json"), "w") as f:
                json.dump(history, f, indent=4)

    print("\nTraining Complete.")
    print(f"Best Test Relative L2 Error: {best_rel_l2:.6f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # 6. Plot Loss Curves
    epochs_range = range(len(history["train_rel_l2"]))
    plt.figure(figsize=(8, 8))
    
    # Plot Rel L2
    plt.semilogy(epochs_range, history["train_rel_l2"], label='Train Rel L2')
    plt.semilogy(epochs_range, history["test_rel_l2"], label='Test Rel L2', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error')
    plt.title('FNO 2D Poisson: Relative L2 Error Convergence')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    loss_plot_path = os.path.join(results_dir, "poisson_fno_losses.png")
    plt.savefig(loss_plot_path, dpi=150)
    print(f"Loss curves saved to: {loss_plot_path}")

if __name__ == "__main__":
    main()
