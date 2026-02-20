"""Train a Flax FNO to learn the Poisson solution operator in 3D (f -> u)."""
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import time
import matplotlib.pyplot as plt
import json

from scirex.operators.fno.models.fno3d import FNO3D
from scirex.training.train_state import create_train_state, TrainState
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse, lp_loss
from scirex.data.datasets.poisson_3d import generator as poisson3d_generator
from experiments.configs.poisson.fno_config import FNO3DConfig


def make_schedule(config: FNO3DConfig):
    """Create learning rate schedule (StepLR equivalent)."""
    scales = {}
    decay_steps = config.scheduler_step_size * config.steps_per_epoch
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
    return schedule

def main():
    # 1. Load Configuration
    config = FNO3DConfig()
    
    # Prng Key
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    
    # 2. Initialize Model
    print(f"Initializing FNO3D (width={config.width}, modes={config.modes_x}x{config.modes_y}x{config.modes_z})...")
    model = FNO3D(
        width=config.width, 
        depth=config.depth, 
        modes_x=config.modes_x, 
        modes_y=config.modes_y, 
        modes_z=config.modes_z,
        out_channels=config.output_channels
    )
    
    # 3. Initialize Optimizer & Scheduler
    schedule = make_schedule(config)
    total_steps = config.epochs * config.steps_per_epoch
    
    # Create Train State
    input_shape = (config.batch_size, config.nx, config.ny, config.nz, config.input_channels)
    state = create_train_state(
        rng=init_rng, 
        model=model, 
        input_shape=input_shape, 
        learning_rate=schedule, 
        weight_decay=config.weight_decay
    )
    
    # 4. Data Generators
    train_gen = poisson3d_generator(
        num_batches=total_steps + 100,
        batch_size=config.batch_size, 
        nx=config.nx, 
        ny=config.ny, 
        nz=config.nz,
        include_mesh=config.include_mesh,
        rng_seed=config.seed
    )
    
    # Test set (fixed seed)
    f_test, u_test = next(poisson3d_generator(
        num_batches=1, 
        batch_size=config.batch_size,
        nx=config.nx, 
        ny=config.ny, 
        nz=config.nz,
        include_mesh=config.include_mesh,
        rng_seed=999 
    ))
    test_batch = {"x": jnp.asarray(f_test), "y": jnp.asarray(u_test)}
    
    # Paths
    ckpt_dir = os.path.join(project_root, "experiments/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "poisson3d_fno_params.pkl")

    results_dir = os.path.join(project_root, "experiments/results/poisson3d")
    os.makedirs(results_dir, exist_ok=True)

    best_rel_l2 = float("inf")
    history = {
        "train_mse": [],
        "train_rel_l2": [],
        "test_mse": [],
        "test_rel_l2": []
    }

    # 5. Training Loop
    print(f"Starting training for {config.epochs} epochs ({total_steps} steps)...")
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for _ in range(config.steps_per_epoch):
            f_np, u_np = next(train_gen)
            batch = {"x": jnp.asarray(f_np), "y": jnp.asarray(u_np)}
            state, metrics = train_step(state, batch, mse)
            epoch_loss += float(metrics["loss"])
            
        epoch_time = time.time() - epoch_start_time
        avg_train_mse = epoch_loss / config.steps_per_epoch
        
        # Performance monitoring
        train_metrics_l2 = eval_step(state, batch, lp_loss)
        avg_train_l2 = float(train_metrics_l2["loss"])

        test_metrics_mse = eval_step(state, test_batch, mse)
        test_metrics_l2 = eval_step(state, test_batch, lp_loss)
        v_test_mse = float(test_metrics_mse["loss"])
        v_test_l2 = float(test_metrics_l2["loss"])

        history["train_mse"].append(avg_train_mse)
        history["train_rel_l2"].append(avg_train_l2)
        history["test_mse"].append(v_test_mse)
        history["test_rel_l2"].append(v_test_l2)
        
        if v_test_l2 < best_rel_l2:
            best_rel_l2 = v_test_l2
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(state.params))
        
        current_lr = schedule(state.step)
        
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:4d} | "
                  f"Train MSE: {avg_train_mse:.6e} | "
                  f"Test MSE: {v_test_mse:.6e} | "
                  f"Test Rel L2: {v_test_l2:.6f} | "
                  f"Best Rel L2: {best_rel_l2:.6f} | "
                  f"LR: {float(current_lr):.2e} | "
                  f"Time: {epoch_time:.2f}s")
            
            with open(os.path.join(results_dir, "fno3d_metrics.json"), "w") as f:
                json.dump(history, f, indent=4)

    print("\nTraining Complete.")
    print(f"Best Test Relative L2 Error: {best_rel_l2:.6f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # 6. Plot Loss Curves
    epochs_range = range(len(history["train_mse"]))
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(epochs_range, history["train_mse"], label='Train MSE')
    plt.semilogy(epochs_range, history["test_mse"], label='Test MSE', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('3D Poisson Loss (MSE)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.semilogy(epochs_range, history["train_rel_l2"], label='Train Rel L2')
    plt.semilogy(epochs_range, history["test_rel_l2"], label='Test Rel L2', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Relative L2 Error')
    plt.title('3D Poisson Accuracy (Rel L2)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "poisson3d_fno_losses.png"), dpi=150)

if __name__ == "__main__":
    main()
