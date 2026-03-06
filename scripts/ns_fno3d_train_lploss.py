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
Training script for Navier-Stokes 2D equation using FNO3D (JAX/Flax).

Dataset: nsforcing from neuraloperator Zenodo archive.
  - x: vorticity input (forcing), y: vorticity output (solution)
  - Each sample is a 2D field on a unit square grid.

Reference: https://github.com/neuraloperator/neuraloperator/blob/main/scripts/train_navier_stokes.py
  - The reference uses GINO3D for irregular data; since our data is on a regular
    unit square grid, we use FNO3D with a trivial z-dimension.
  - Training loss: H1 loss (reference default), with LpLoss and MSE tracked.
  - Evaluation: both H1, relative L2, and MSE.
  - Normalization: channel-wise Gaussian (reference default).

Usage
-----
    python scripts/ns_fno3d_train.py
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys

# Ensure project root is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import json

import torch
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax

from flax import linen as nn

from scirex.operators.models import FNO3D
from scirex.operators.training import create_train_state, GaussianNormalizer
from scirex.operators.losses.data_losses import h1_loss, lp_loss, mse
from configs.ns_fno3d_config import NSFNO3DConfig


# LR SCHEDULE
def make_schedule(config: NSFNO3DConfig):
    """Create learning rate schedule from config."""
    spe = config.steps_per_epoch
    total_steps = config.epochs * spe

    if config.scheduler_type == "step":
        # StepLR: decay by gamma every scheduler_step_size epochs
        decay_steps = config.scheduler_step_size * spe
        lr_schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=decay_steps,
            decay_rate=config.scheduler_gamma,
            staircase=True,
        )
    elif config.scheduler_type == "cosine":
        warmup_steps = min(310, total_steps // 10)
        cosine_decay_steps = max(config.cosine_decay_epochs * spe - warmup_steps, 1)
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=cosine_decay_steps,
            alpha=0.0,
        )
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
                cosine_schedule,
            ],
            boundaries=[warmup_steps],
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {config.scheduler_type}")

    return lr_schedule

# DATA LOADING
def load_ns_data(config: NSFNO3DConfig):
    """Load Navier-Stokes .pt files (dict with 'x' and 'y' keys).

    Returns numpy arrays with shape ``(N, H, W, 1, 1)`` ready for FNO3D.
    """
    print(f"Loading data from {config.data_dir}/{config.train_file}...")

    train_dict = torch.load(
        os.path.join(config.data_dir, config.train_file), map_location="cpu"
    )
    test_dict = torch.load(
        os.path.join(config.data_dir, config.test_file), map_location="cpu"
    )

    # Raw shapes: (N, H, W) -> take n_train/n_test samples
    train_x = train_dict["x"][: config.n_train].numpy()
    train_y = train_dict["y"][: config.n_train].numpy()
    test_x = test_dict["x"][: config.n_test].numpy()
    test_y = test_dict["y"][: config.n_test].numpy()

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

# JIT-COMPILED STEP FUNCTIONS
@jax.jit
def train_step_h1(state, batch_x, batch_y):
    """Train step using H1 loss (reference default)."""

    def loss_fn(params):
        pred = state.apply_fn({"params": params}, batch_x)
        loss = h1_loss(pred, batch_y)
        rel_l2 = lp_loss(pred, batch_y)
        return loss, rel_l2

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, rel_l2), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "rel_l2": rel_l2}


@jax.jit
def train_step_lp(state, batch_x, batch_y):
    """Train step using LpLoss (relative L2)."""

    def loss_fn(params):
        pred = state.apply_fn({"params": params}, batch_x)
        return lp_loss(pred, batch_y)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "rel_l2": loss}


@jax.jit
def eval_step(state, batch_x, batch_y):
    """Evaluate: compute H1, relative L2, and MSE."""
    pred = state.apply_fn({"params": state.params}, batch_x)
    return {
        "h1": h1_loss(pred, batch_y),
        "rel_l2": lp_loss(pred, batch_y),
        "mse": mse(pred, batch_y),
        "pred": pred,
    }


# MAIN
def main():
    config = NSFNO3DConfig()
    print(f"JAX devices: {jax.devices()}")

    # ── 1. Load Data ──
    train_x, train_y, test_x, test_y = load_ns_data(config)

    # ── 2. Normalize ──
    if config.encode_output:
        y_normalizer = GaussianNormalizer(train_y)
        train_y = y_normalizer.encode(train_y)
        test_y = y_normalizer.encode(test_y)
        print("  Output normalized (channel-wise Gaussian)")

    if config.encode_input:
        x_normalizer = GaussianNormalizer(train_x)
        train_x = x_normalizer.encode(train_x)
        test_x = x_normalizer.encode(test_x)
        print("  Input normalized (channel-wise Gaussian)")

    # Convert to jnp
    train_x, train_y = jnp.array(train_x), jnp.array(train_y)
    test_x, test_y = jnp.array(test_x), jnp.array(test_y)

    # ── 3. Model ──
    print(
        f"Initializing FNO3D (hidden_channels={config.hidden_channels}, "
        f"modes={config.n_modes}, layers={config.n_layers})..."
    )
    model = FNO3D(
        hidden_channels=config.hidden_channels,
        n_layers=config.n_layers,
        n_modes=config.n_modes,
        out_channels=config.out_channels,
        lifting_channel_ratio=config.lifting_channel_ratio,
        projection_channel_ratio=config.projection_channel_ratio,
        use_grid=config.use_grid,
        use_norm=config.use_norm,
        fno_skip=config.fno_skip,
        channel_mlp_skip=config.channel_mlp_skip,
        use_channel_mlp=config.use_channel_mlp,
        padding=config.domain_padding,
    )

    # ── 4. Optimizer + Train State ──
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    input_shape = train_x[:1].shape
    schedule = make_schedule(config)

    state = create_train_state(
        rng=init_rng,
        model=model,
        input_shape=input_shape,
        learning_rate=schedule,
        weight_decay=config.weight_decay,
    )

    n_params = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"  Model parameters: {n_params:,}")

    # ── 5. Select train step function ──
    step_fn = train_step_lp
    config.train_loss = "lp"  # Explicitly set for accurate logging/naming

    # ── 6. Prepare output dirs ──
    results_dir = os.path.join(project_root, config.results_dir)
    ckpt_dir = os.path.join(project_root, config.checkpoint_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, config.model_name)

    # ── 7. Training Loop ──
    print(f"\nStarting training for {config.epochs} epochs (loss={config.train_loss})...")
    print(
        f"{'Epoch':>6} | {'Train Loss':>12} {'Train RelL2':>12} | "
        f"{'Test H1':>10} {'Test RelL2':>10} | {'Time':>8}"
    )
    print("-" * 85)

    num_train = train_x.shape[0]
    num_test = test_x.shape[0]
    best_test_l2 = float("inf")
    history = {"train_rel_l2": [], "test_rel_l2": [], "test_h1": []}
    rng_key = jax.random.PRNGKey(config.seed + 1)

    for epoch in range(1, config.epochs + 1):
        # Shuffle training data
        rng_key, shuffle_key = jax.random.split(rng_key)
        perm = jax.random.permutation(shuffle_key, num_train)
        shuffled_x = train_x[perm]
        shuffled_y = train_y[perm]

        epoch_loss = 0.0
        epoch_rel_l2 = 0.0
        n_batches = 0

        start_time = time.time()
        for i in range(0, num_train, config.batch_size):
            end_i = i + config.batch_size
            if end_i > num_train:
                continue  # skip incomplete batch

            bx = shuffled_x[i:end_i]
            by = shuffled_y[i:end_i]
            state, metrics = step_fn(state, bx, by)
            epoch_loss += float(metrics["loss"])
            epoch_rel_l2 += float(metrics["rel_l2"])
            n_batches += 1

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_rel_l2 = epoch_rel_l2 / max(n_batches, 1)

        # Evaluate on test set
        test_h1_total = 0.0
        test_l2_total = 0.0
        n_test_batches = 0
        for i in range(0, num_test, config.batch_size):
            end_i = min(i + config.batch_size, num_test)
            bx = test_x[i:end_i]
            by = test_y[i:end_i]
            ev = eval_step(state, bx, by)
            test_h1_total += float(ev["h1"])
            test_l2_total += float(ev["rel_l2"])
            n_test_batches += 1

        test_h1_avg = test_h1_total / max(n_test_batches, 1)
        test_l2_avg = test_l2_total / max(n_test_batches, 1)

        history["train_rel_l2"].append(avg_rel_l2)
        history["test_rel_l2"].append(test_l2_avg)
        history["test_h1"].append(test_h1_avg)

        # Checkpoint best model
        if test_l2_avg < best_test_l2:
            best_test_l2 = test_l2_avg
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(state.params))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{epoch:4d}] | {avg_loss:12.6f} {avg_rel_l2:12.6f} | "
                f"{test_h1_avg:10.4f} {test_l2_avg:10.4f} | {epoch_time:7.2f}s"
            )
            # Persist metrics incrementally
            with open(os.path.join(results_dir, "ns_fno3d_metrics.json"), "w") as f:
                json.dump(history, f, indent=4)

    print(f"\nTraining finished. Best Test Rel L2: {best_test_l2:.6f}")

    # ── 8. Save final metrics ──
    with open(os.path.join(results_dir, "ns_fno3d_metrics.json"), "w") as f:
        json.dump(history, f, indent=4)
    print(f"Metrics saved to: {results_dir}/ns_fno3d_metrics.json")

    # ── 9. Save sample predictions for visualization ──
    n_viz = min(8, num_test)
    viz_pred = state.apply_fn({"params": state.params}, test_x[:n_viz])
    np.savez(
        os.path.join(results_dir, "ns_fno3d_predictions.npz"),
        inputs=np.array(test_x[:n_viz]),
        targets=np.array(test_y[:n_viz]),
        predictions=np.array(viz_pred),
    )
    print(f"Sample predictions saved to: {results_dir}/ns_fno3d_predictions.npz")
    print("Run `python scripts/ns_fno3d_visualize.py` to generate plots.")


if __name__ == "__main__":
    main()
