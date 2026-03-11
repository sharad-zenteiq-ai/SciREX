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
Train SciREX FNO2D on the neuraloperator 1-D Burgers benchmark dataset.

PDE:        ∂u/∂t  +  u · ∂u/∂x  =  ν · ∂²u/∂x²
Domain:     x ∈ [0, 1]  (periodic),  t ∈ [0, T],  ν = 0.01
Operator:   u₀(x)  ──FNO2D──>  u(x, T_final)

Dataset     (neuraloperator .pt files, confirmed structure)
-----------
    burgers_train_16.pt  →  x:(1200,16)  y:(1200,16,17)  visc:0.01
    burgers_test_16.pt   →  x:(1200,16)  y:(1200,16,17)  visc:0.01

FNO2D treatment
---------------
The data is 1-D but we use FNO2D by adding a trivial spatial dimension ny=1:
    Input  (N, 16, 1, 2)  →  [u₀(x),  x_grid]
    Target (N, 16, 1, 1)  →  u(x, T_final)

Pipeline (mirrors train_poisson2d_fno.py and train_darcy_pt_fno2d.py)
----------------------------------------------------------------------
  1  Config        → BurgersPtFNO2DConfig
  2  Data          → load_burgers_pt  (burgers_pt.py)
  3  Normalisation → GaussianNormalizer  (encode both x and y)
  4  Model         → FNO2D  (SciREX JAX/Flax)
  5  TrainState    → create_train_state  (AdamW + grad clip)
  6  LR schedule   → warmup + cosine decay
  7  Training loop → JIT train step, epoch shuffle, checkpoint
  8  Plots         → 5 figures saved to experiments/results/burgers_pt_fno2d/

Outputs saved to:  experiments/results/burgers_pt_fno2d/
    loss_curves.png            — semi-log train/test Rel-L2 vs epoch
    field_best_sample.png      — IC | GT | Pred | Error for best test sample
    field_worst_sample.png     — same for worst test sample
    error_histogram.png        — histogram of per-sample test Rel-L2
    multi_sample_comparison.png— 6-row grid of random test samples

Run:
    cd /path/to/SciREX-operators
    python scripts/train_burgers_pt_fno2d.py

    # Override dataset paths without editing config:
    BURGERS_TRAIN_PATH=/my/path/burgers_train_16.pt  \\
    BURGERS_TEST_PATH=/my/path/burgers_test_16.pt    \\
    python scripts/train_burgers_pt_fno2d.py
"""

import os
import sys
import json
import time

# ── Deterministic GPU ops (before JAX import) ─────────────────────────────────
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# ── Project root on sys.path ──────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax import linen as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scirex.operators.models.fno import FNO2D
from scirex.operators.training import create_train_state, GaussianNormalizer
from scirex.operators.losses import lp_loss
from scirex.operators.data.burgers_pt import load_burgers_pt
from configs.burgers_pt_fno_config import BurgersPtFNO2DConfig


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule  (identical helper to all other SciREX training scripts)
# ─────────────────────────────────────────────────────────────────────────────

def make_schedule(config: BurgersPtFNO2DConfig):
    """Linear warmup  +  cosine or step decay."""
    spe          = getattr(config, "_steps_per_epoch", config.steps_per_epoch)
    total_steps  = config.epochs * spe
    warmup_steps = min(310, total_steps // 10)

    if config.scheduler_type == "cosine":
        decay_steps = max(config.cosine_decay_epochs * spe - warmup_steps, 1)
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
                optax.cosine_decay_schedule(
                    init_value=config.learning_rate,
                    decay_steps=decay_steps,
                    alpha=0.0,
                ),
            ],
            boundaries=[warmup_steps],
        )
    elif config.scheduler_type == "step":
        scales        = {}
        step_size     = config.scheduler_step_size * spe
        current_scale = 1.0
        for i in range(1, config.epochs // config.scheduler_step_size + 1):
            current_scale       *= config.scheduler_gamma
            scales[i * step_size] = current_scale
        schedule = optax.piecewise_constant_schedule(
            init_value=config.learning_rate,
            boundaries_and_scales=scales,
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {config.scheduler_type!r}")
    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, save_path: str) -> None:
    """Semi-log convergence plot of train and test Rel-L2."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = range(len(history["train_rel_l2"]))
    ax.semilogy(ep, history["train_rel_l2"], lw=2, label="Train  Rel-L2")
    ax.semilogy(ep, history["test_rel_l2"],  lw=2, ls="--", label="Test   Rel-L2")
    ax.set_xlabel("Epoch",               fontsize=12)
    ax.set_ylabel("Relative L2 Error",   fontsize=12)
    ax.set_title(
        "FNO2D — 1-D Burgers (neuraloperator dataset):\nConvergence",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="-", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve       → {save_path}")


def plot_1d_comparison(
    x_sample: np.ndarray,    # (nx, 1, 2)  — first channel is u0
    y_true:   np.ndarray,    # (nx, 1, 1)
    y_pred:   np.ndarray,    # (nx, 1, 1)
    title:    str,
    save_path: str,
) -> None:
    """
    Two-panel 1-D line plot:
        Left  — IC u₀(x) vs predicted u(x,T) vs ground-truth u(x,T)
        Right — absolute pointwise error |u_GT − u_pred|
    """
    nx   = x_sample.shape[0]
    xgrid = x_sample[:, 0, 1]          # grid_x channel
    u0    = x_sample[:, 0, 0]          # IC channel
    u_gt  = y_true[:, 0, 0]
    u_pr  = y_pred[:, 0, 0]
    err   = np.abs(u_gt - u_pr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: overlay IC, GT, prediction
    axes[0].plot(xgrid, u0,   lw=1.5, ls=":", color="grey",      label="IC  u₀(x)")
    axes[0].plot(xgrid, u_gt, lw=2,   color="steelblue",          label="GT  u(x,T)")
    axes[0].plot(xgrid, u_pr, lw=2,   ls="--", color="tomato",    label="FNO u(x,T)")
    axes[0].set_xlabel("x", fontsize=11)
    axes[0].set_ylabel("u", fontsize=11)
    axes[0].set_title("IC vs Ground Truth vs Prediction", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Right: absolute error
    axes[1].fill_between(xgrid, err, alpha=0.4, color="tomato")
    axes[1].plot(xgrid, err, lw=2, color="tomato")
    axes[1].set_xlabel("x", fontsize=11)
    axes[1].set_ylabel("|u_GT − u_pred|", fontsize=11)
    axes[1].set_title("Absolute Pointwise Error", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Field comparison → {save_path}")


def plot_error_histogram(
    y_true:   np.ndarray,    # (N, nx, 1, 1)
    y_pred:   np.ndarray,    # (N, nx, 1, 1)
    save_path: str,
) -> None:
    """Histogram of per-sample relative L2 errors on the test set."""
    n      = y_true.shape[0]
    diff   = (y_pred - y_true).reshape(n, -1)
    target = y_true.reshape(n, -1)
    rel_l2 = np.linalg.norm(diff, axis=1) / (np.linalg.norm(target, axis=1) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rel_l2, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(rel_l2),   color="red",    lw=2, ls="--",
               label=f"Mean   {np.mean(rel_l2):.4f}")
    ax.axvline(np.median(rel_l2), color="orange", lw=2, ls=":",
               label=f"Median {np.median(rel_l2):.4f}")
    ax.set_xlabel("Per-sample Relative L2 Error", fontsize=12)
    ax.set_ylabel("Count",                         fontsize=12)
    ax.set_title("FNO2D Burgers — Test-set Error Distribution", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Error histogram  → {save_path}")


def plot_multi_sample(
    x_test:   np.ndarray,    # (N, nx, 1, 2)
    y_test:   np.ndarray,    # (N, nx, 1, 1)
    y_pred:   np.ndarray,    # (N, nx, 1, 1)
    n_samples: int,
    save_path: str,
) -> None:
    """
    Grid of n_samples rows, 3 columns:
        col 0 — IC u₀(x)
        col 1 — ground truth u(x,T) vs FNO prediction (overlaid)
        col 2 — absolute error
    """
    n_samples = min(n_samples, x_test.shape[0])
    fig, axes = plt.subplots(
        n_samples, 3,
        figsize=(12, 3.0 * n_samples),
        sharex=True,
    )
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["IC  u₀(x)",  "GT vs Prediction",  "Abs error"]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=11, fontweight="bold")

    nx    = x_test.shape[1]
    xgrid = x_test[0, :, 0, 1]      # normalised x coordinate

    for row in range(n_samples):
        u0   = x_test[row, :, 0, 0]
        u_gt = y_test[row, :, 0, 0]
        u_pr = y_pred[row, :, 0, 0]
        err  = np.abs(u_gt - u_pr)

        # IC
        axes[row, 0].plot(xgrid, u0, color="grey", lw=1.5)
        axes[row, 0].set_ylabel(f"Sample {row+1}", fontsize=9)
        axes[row, 0].grid(True, alpha=0.3)

        # Overlay GT and prediction
        axes[row, 1].plot(xgrid, u_gt, lw=2, color="steelblue", label="GT")
        axes[row, 1].plot(xgrid, u_pr, lw=2, ls="--", color="tomato", label="FNO")
        if row == 0:
            axes[row, 1].legend(fontsize=8, loc="upper right")
        axes[row, 1].grid(True, alpha=0.3)

        # Absolute error
        axes[row, 2].fill_between(xgrid, err, alpha=0.35, color="tomato")
        axes[row, 2].plot(xgrid, err, color="tomato", lw=1.5)
        axes[row, 2].grid(True, alpha=0.3)

    for col in range(3):
        axes[-1, col].set_xlabel("x", fontsize=10)

    fig.suptitle("Burgers (ν=0.01) — FNO2D Multi-sample Predictions", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Multi-sample     → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Config ─────────────────────────────────────────────────────────────
    config = BurgersPtFNO2DConfig()

    # Allow environment-variable overrides
    config.train_path = os.environ.get("BURGERS_TRAIN_PATH", config.train_path)
    config.test_path  = os.environ.get("BURGERS_TEST_PATH",  config.test_path)

    print("=" * 65)
    print("  SciREX FNO2D — 1-D Burgers (neuraloperator dataset)")
    print("=" * 65)
    print(f"  train_path : {config.train_path}")
    print(f"  test_path  : {config.test_path}")
    print(f"  nx={config.nx}  ny={config.ny}  (trivial dim for FNO2D)")
    print(f"  n_train={config.n_train}  n_test={config.n_test}  "
          f"epochs={config.epochs}  batch={config.batch_size}")
    print(f"  FNO modes  : {config.n_modes}  hidden : {config.hidden_channels}  "
          f"layers : {config.n_layers}")
    print()

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    # ── 2. Data ───────────────────────────────────────────────────────────────
    x_train, y_train, x_test, y_test = load_burgers_pt(
        train_path=config.train_path,
        test_path=config.test_path,
        n_train=config.n_train,
        n_test=config.n_test,
    )

    # Shapes: (N, nx, 1, 2) and (N, nx, 1, 1)
    nx         = x_train.shape[1]
    ny         = x_train.shape[2]     # 1
    in_ch      = x_train.shape[-1]    # 2

    x_train_jnp = jnp.asarray(x_train)
    y_train_jnp = jnp.asarray(y_train)
    x_test_jnp  = jnp.asarray(x_test)
    y_test_jnp  = jnp.asarray(y_test)

    # ── 3. Normalisation ──────────────────────────────────────────────────────
    # GaussianNormalizer reduces over (N, nx, ny) and keeps channel dim
    x_norm = GaussianNormalizer(x_train_jnp) if config.encode_input  else None
    y_norm = GaussianNormalizer(y_train_jnp) if config.encode_output else None

    x_tr = x_norm.encode(x_train_jnp) if x_norm else x_train_jnp
    y_tr = y_norm.encode(y_train_jnp) if y_norm else y_train_jnp
    x_te = x_norm.encode(x_test_jnp)  if x_norm else x_test_jnp

    test_batch_enc = {
        "x": x_te,
        "y": y_norm.encode(y_test_jnp) if y_norm else y_test_jnp,
    }

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print(f"Building FNO2D: modes={config.n_modes}  hidden={config.hidden_channels}"
          f"  layers={config.n_layers}  use_norm={config.use_norm}")
    model = FNO2D(
        hidden_channels=config.hidden_channels,
        n_layers=config.n_layers,
        n_modes=config.n_modes,
        out_channels=config.out_channels,
        lifting_channel_ratio=config.lifting_channel_ratio,
        projection_channel_ratio=config.projection_channel_ratio,
        use_grid=False,               # grid already in channel 1
        use_norm=config.use_norm,
        fno_skip=config.fno_skip,
        channel_mlp_skip=config.channel_mlp_skip,
        use_channel_mlp=config.use_channel_mlp,
        padding=config.domain_padding,
        activation=nn.gelu,
    )

    # ── 5. Optimiser / schedule ───────────────────────────────────────────────
    actual_n_train  = x_train.shape[0]
    steps_per_epoch = actual_n_train // config.batch_size
    config._steps_per_epoch = steps_per_epoch     # used by make_schedule
    schedule    = make_schedule(config)
    total_steps = config.epochs * steps_per_epoch

    state = create_train_state(
        rng=init_rng,
        model=model,
        input_shape=(config.batch_size, nx, ny, in_ch),
        learning_rate=schedule,
        weight_decay=config.weight_decay,
    )

    # ── 6. Paths ──────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(project_root, "experiments", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "burgers_pt_fno2d_best.pkl")

    results_dir = os.path.join(project_root, "experiments", "results", "burgers_pt_fno2d")
    os.makedirs(results_dir, exist_ok=True)

    best_rel_l2 = float("inf")
    history = {"train_rel_l2": [], "test_rel_l2": []}

    # ── 7. JIT train step ─────────────────────────────────────────────────────
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            pred = state.apply_fn({"params": params}, batch["x"])
            return lp_loss(pred, batch["y"])
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), {"loss": loss}

    # ── 8. Training loop ──────────────────────────────────────────────────────
    print(f"\nTraining: {config.epochs} epochs  ×  {steps_per_epoch} steps"
          f" = {total_steps} total updates\n")
    rng_key = jax.random.PRNGKey(config.seed + 1)

    for epoch in range(config.epochs):
        t0         = time.time()
        epoch_loss = 0.0

        # Shuffle training data every epoch
        rng_key, sk = jax.random.split(rng_key)
        perm   = jax.random.permutation(sk, actual_n_train)
        x_shuf = x_tr[perm]
        y_shuf = y_tr[perm]

        for step in range(steps_per_epoch):
            s = step * config.batch_size
            e = s    + config.batch_size
            state, metrics = train_step(
                state, {"x": x_shuf[s:e], "y": y_shuf[s:e]}
            )
            epoch_loss += float(metrics["loss"])

        avg_train = epoch_loss / steps_per_epoch

        # ── Evaluation ────────────────────────────────────────────────────
        pred_enc = state.apply_fn({"params": state.params}, test_batch_enc["x"])
        pred_dec = y_norm.decode(pred_enc) if y_norm else pred_enc
        v_test   = float(lp_loss(pred_dec, y_test_jnp))

        history["train_rel_l2"].append(avg_train)
        history["test_rel_l2"].append(v_test)

        # Checkpoint on improvement
        if v_test < best_rel_l2:
            best_rel_l2 = v_test
            with open(ckpt_path, "wb") as fp:
                fp.write(flax.serialization.to_bytes(state.params))

        # Console log every 10 epochs
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            lr_now = float(schedule(state.step))
            print(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train Rel-L2: {avg_train:.6e} | "
                f"Test Rel-L2: {v_test:.6f} | "
                f"Best: {best_rel_l2:.6f} | "
                f"LR: {lr_now:.2e} | "
                f"Time: {time.time()-t0:.2f}s"
            )
            with open(os.path.join(results_dir, "metrics.json"), "w") as fp:
                json.dump(history, fp, indent=4)

    print(f"\n{'='*65}")
    print(f"  Training complete.  Best Test Rel-L2 = {best_rel_l2:.6f}")
    print(f"  Checkpoint → {ckpt_path}")
    print(f"{'='*65}\n")

    # ── 9. Load best checkpoint for evaluation ────────────────────────────────
    with open(ckpt_path, "rb") as fp:
        best_params = flax.serialization.from_bytes(state.params, fp.read())

    # Full test-set predictions (decoded to physical units)
    n_test_actual = x_test.shape[0]
    y_pred_all = []
    for i in range(0, n_test_actual, config.batch_size):
        chunk = x_te[i : i + config.batch_size]
        p_enc = state.apply_fn({"params": best_params}, chunk)
        p_dec = y_norm.decode(p_enc) if y_norm else p_enc
        y_pred_all.append(np.array(p_dec))
    y_pred_np = np.concatenate(y_pred_all, axis=0)   # (n_test, nx, 1, 1)
    y_test_np = np.array(y_test_jnp)

    # Final per-sample metrics
    diff_all   = (y_pred_np - y_test_np).reshape(n_test_actual, -1)
    targ_all   = y_test_np.reshape(n_test_actual, -1)
    rel_l2_all = np.linalg.norm(diff_all, axis=1) / (np.linalg.norm(targ_all, axis=1) + 1e-8)

    print("Final test-set metrics (best checkpoint):")
    print(f"  Mean Rel-L2  : {rel_l2_all.mean():.6f}")
    print(f"  Median Rel-L2: {np.median(rel_l2_all):.6f}")
    print(f"  Max Rel-L2   : {rel_l2_all.max():.6f}")
    print(f"  Min Rel-L2   : {rel_l2_all.min():.6f}\n")

    # ── 10. Plots ──────────────────────────────────────────────────────────────
    print("Saving plots …")

    # (a) Loss curves
    plot_loss_curves(
        history,
        os.path.join(results_dir, "loss_curves.png"),
    )

    # (b) Best sample (lowest error)
    best_idx = int(np.argmin(rel_l2_all))
    plot_1d_comparison(
        x_test[best_idx],
        y_test_np[best_idx],
        y_pred_np[best_idx],
        title=f"Best prediction  (sample {best_idx},  Rel-L2={rel_l2_all[best_idx]:.4f})",
        save_path=os.path.join(results_dir, "field_best_sample.png"),
    )

    # (c) Worst sample (highest error)
    worst_idx = int(np.argmax(rel_l2_all))
    plot_1d_comparison(
        x_test[worst_idx],
        y_test_np[worst_idx],
        y_pred_np[worst_idx],
        title=f"Worst prediction  (sample {worst_idx},  Rel-L2={rel_l2_all[worst_idx]:.4f})",
        save_path=os.path.join(results_dir, "field_worst_sample.png"),
    )

    # (d) Error histogram
    plot_error_histogram(
        y_test_np, y_pred_np,
        os.path.join(results_dir, "error_histogram.png"),
    )

    # (e) Multi-sample comparison (6 random samples)
    np.random.seed(config.seed)
    sample_ids = np.random.choice(n_test_actual, size=min(6, n_test_actual), replace=False)
    plot_multi_sample(
        x_test[sample_ids],
        y_test_np[sample_ids],
        y_pred_np[sample_ids],
        n_samples=len(sample_ids),
        save_path=os.path.join(results_dir, "multi_sample_comparison.png"),
    )

    print(f"\nAll outputs saved under: {results_dir}")


if __name__ == "__main__":
    main()
