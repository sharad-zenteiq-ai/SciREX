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
Training script for Navier-Stokes 2D equation using FNO2D (JAX/Flax).
Aligned with the official neuraloperator reference.

Reference: https://github.com/neuraloperator/neuraloperator/blob/main/scripts/train_navier_stokes.py
   - Default Training loss: H1Loss
   - Default Optimizer: AdamW with StepLR
   - Framework: JAX/Flax
"""

import os
import sys
import time
import json
from functools import partial
import torch
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import linen as nn

# Ensure project root is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scirex.operators.models.fno import FNO
from scirex.operators.training import create_train_state, GaussianNormalizer
from scirex.operators.losses.data_losses import h1_loss, lp_loss, mse
from configs.ns_fno2d_config import NSFNO2DConfig

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ── 1. LR SCHEDULE ──
def make_schedule(config: NSFNO2DConfig):
    """Create learning rate schedule from config."""
    spe = config.get_steps_per_epoch()
    
    if config.opt.scheduler == "StepLR":
        decay_steps = config.opt.step_size * spe
        return optax.exponential_decay(
            init_value=config.opt.learning_rate,
            transition_steps=decay_steps,
            decay_rate=config.opt.gamma,
            staircase=True,
        )
    elif config.scheduler_type == "cosine":
        total_steps = config.opt.n_epochs * spe
        return optax.cosine_decay_schedule(
            init_value=config.opt.learning_rate,
            decay_steps=total_steps,
            alpha=0.0,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.opt.scheduler}")

# ── 2. DATA LOADING ──
def load_ns_data(config: NSFNO2DConfig):
    """Load Navier-Stokes .pt files and prepare for FNO2D."""
    data_path = config.data.folder
    print(f"Loading data from {data_path}...")

    train_dict = torch.load(os.path.join(data_path, config.data.train_file), map_location="cpu")
    test_dict = torch.load(os.path.join(data_path, config.data.test_file), map_location="cpu")

    train_x = train_dict["x"][: config.data.n_train].numpy()[..., np.newaxis]
    train_y = train_dict["y"][: config.data.n_train].numpy()[..., np.newaxis]
    test_x = test_dict["x"][: config.data.n_tests[0]].numpy()[..., np.newaxis]
    test_y = test_dict["y"][: config.data.n_tests[0]].numpy()[..., np.newaxis]

    print(f"  Train: x={train_x.shape}, y={train_y.shape}")
    print(f"  Test:  x={test_x.shape}, y={test_y.shape}")

    return train_x, train_y, test_x, test_y

# ── 3. STEP FUNCTIONS ──

@partial(jax.jit, static_argnames=("loss_type",))
def train_step(state, batch_x, batch_y, loss_type="h1"):
    """Generic train step supporting H1 and L2 losses."""
    def loss_fn(params):
        pred = state.apply_fn({"params": params}, batch_x)
        if loss_type == "h1":
            return h1_loss(pred, batch_y)
        else:
            return lp_loss(pred, batch_y)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss}

@jax.jit
def eval_step(state, batch_x, batch_y_raw, y_mean, y_std):
    """Eval step: results in physical scale."""
    pred_encoded = state.apply_fn({"params": state.params}, batch_x)
    pred = pred_encoded * y_std + y_mean

    return {
        "h1": h1_loss(pred, batch_y_raw),
        "l2": lp_loss(pred, batch_y_raw),
        "mse": mse(pred, batch_y_raw),
    }

# ── 4. MAIN ──
def main():
    config = NSFNO2DConfig()
    print(f"Config: {config.opt.training_loss} loss, {config.opt.n_epochs} epochs")

    # Load & Prepare Data
    train_x, train_y, test_x, test_y = load_ns_data(config)
    test_y_raw = jnp.array(test_y)

    # Normalization (Reference DataProcessor style)
    y_normalizer = GaussianNormalizer(train_y)
    train_y_norm = jnp.array(y_normalizer.encode(train_y))
    y_mean, y_std = jnp.array(y_normalizer.mean), jnp.array(y_normalizer.std)

    x_normalizer = GaussianNormalizer(train_x)
    train_x_norm = jnp.array(x_normalizer.encode(train_x))
    test_x_norm = jnp.array(x_normalizer.encode(test_x))

    # Model
    model = FNO(
        hidden_channels=config.model.hidden_channels,
        n_layers=config.model.n_layers,
        n_modes=config.model.n_modes,
        out_channels=config.model.out_channels,
        lifting_channel_ratio=config.model.lifting_channel_ratio,
        projection_channel_ratio=config.model.projection_channel_ratio,
        use_grid=getattr(config.model, "use_grid", True),
        use_norm=(config.model.use_norm != "None"),
        fno_skip=config.model.fno_skip,
        padding=config.model.domain_padding,
        use_channel_mlp=config.model.use_channel_mlp,
    )

    # Optimizer & Scheduler
    schedule = make_schedule(config)
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(
        rng=init_rng,
        model=model,
        input_shape=train_x_norm[:1].shape,
        learning_rate=schedule,
        weight_decay=config.opt.weight_decay,
    )

    # Training Loop
    print(f"\nTraining {config.opt.training_loss}...")
    print("Pre-compiling training and eval steps...")
    num_train = train_x.shape[0]
    n_batches = max(num_train // config.data.batch_size, 1)
    best_test_l2 = float("inf")
    best_train_loss = float("inf")
    
    total_start_time = time.time()
    for epoch in range(1, config.opt.n_epochs + 1):
        epoch_loss = 0.0
        start_time = time.time()
        
        # Shuffle for each epoch
        rng, shuffle_rng = jax.random.split(rng)
        idx = jax.random.permutation(shuffle_rng, num_train)
        
        for i in range(0, num_train, config.data.batch_size):
            batch_idx = idx[i : i + config.data.batch_size]
            bx, by = train_x_norm[batch_idx], train_y_norm[batch_idx]
            state, metrics = train_step(state, bx, by, loss_type=config.opt.training_loss)
            if epoch == 1 and i == 0:
                print("First batch completed (compilation done).")
            epoch_loss += float(metrics["loss"])

        epoch_time = time.time() - start_time
        
        # Test Eval (batched to avoid OOM)
        test_batch_size = config.data.test_batch_sizes[0]
        num_test = test_x_norm.shape[0]
        test_h1, test_l2_val, test_mse_val = 0.0, 0.0, 0.0
        for ti in range(0, num_test, test_batch_size):
            bx_t = test_x_norm[ti : ti + test_batch_size]
            by_t = test_y_raw[ti : ti + test_batch_size]
            m = eval_step(state, bx_t, by_t, y_mean, y_std)
            test_h1 += float(m["h1"])
            test_l2_val += float(m["l2"])
            test_mse_val += float(m["mse"])
        num_batches = num_train // config.data.batch_size
        avg_epoch_loss = epoch_loss / num_train
        train_err = epoch_loss / num_batches
        
        if epoch % 1 == 0 or epoch == 1:
            # Format exactly like neuraloperator Trainer.log_training and Trainer.log_eval
            msg_train = f"[{epoch}] time={epoch_time:.2f}, avg_loss={avg_epoch_loss:.4f}, train_err={train_err:.4f}"
            print(msg_train)
            
            # Average the other eval metrics (Mean Relative Error)
            test_h1_avg = test_h1 / num_test
            test_l2_avg = test_l2_val / num_test
            msg_eval = f"Eval: 128_h1={test_h1_avg:.4f}, 128_l2={test_l2_avg:.4f}"
            print(msg_eval)
            sys.stdout.flush()

        if test_l2_avg < best_test_l2:
            best_test_l2 = test_l2_avg
            best_train_loss = avg_epoch_loss

    total_time = time.time() - total_start_time
    print(f"\nTotal Computational Time: {total_time:.2f}s")
    print(f"Final Best Test L2: {best_test_l2:.6f}")
    print(f"Training Loss at Best Test L2: {best_train_loss:.6f}")

if __name__ == "__main__":
    main()
