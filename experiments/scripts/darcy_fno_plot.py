"""Train FNO2D and plot results (prediction vs truth) for the 2D Darcy problem."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scirex.operators.fno.models.fno2d import FNO2D
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse
from scirex.data.datasets.darcy import generator as darcy_generator

def main():
    # Config
    batch_size = 16
    nx, ny, in_ch = 64, 64, 1
    width = 32  # Smaller width for faster execution in plotting demo
    depth = 4
    modes_x = 16
    modes_y = 16
    out_ch = 1
    lr = 1e-3
    rng = jax.random.PRNGKey(42)

    model = FNO2D(
        width=width, 
        depth=depth, 
        modes_x=modes_x, 
        modes_y=modes_y, 
        out_channels=out_ch
    )
    state = create_train_state(rng, model, (batch_size, nx, ny, in_ch), learning_rate=lr)

    # Training generator - Using FNO-style binary permeability
    num_batches = 500
    gen = darcy_generator(
        num_batches=num_batches, 
        batch_size=batch_size, 
        nx=nx, ny=ny, 
        rng_seed=0,
        mode="binary",
        a_low=4.0,
        a_high=12.0
    )

    losses = []
    print(f"Starting training (FNO Darcy plotting, modes={modes_x}x{modes_y})...")
    for step, (a_np, u_np) in enumerate(gen):
        batch = {"x": jnp.asarray(a_np), "y": jnp.asarray(u_np)}
        state, metrics = train_step(state, batch, mse)
        losses.append(float(metrics["loss"]))
        if step % 20 == 0:
            print(f"step {step:4d}, loss: {losses[-1]:.6e}")

    # Inference on a fresh test sample
    a_test, u_test = next(darcy_generator(
        num_batches=1, batch_size=1, nx=nx, ny=ny, 
        rng_seed=999, mode="binary", a_low=4.0, a_high=12.0
    ))
    eval_batch = {"x": jnp.asarray(a_test), "y": jnp.asarray(u_test)}
    out = eval_step(state, eval_batch, mse)
    u_pred = np.array(out["preds"][0, ..., 0])
    u_true = u_test[0, ..., 0]
    a_field = a_test[0, ..., 0]

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im0 = axes[0, 0].imshow(a_field, origin='lower')
    axes[0, 0].set_title("Input Permeability a(x,y)")
    fig.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(u_true, origin='lower')
    axes[0, 1].set_title("True Pressure u(x,y)")
    fig.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(u_pred, origin='lower')
    axes[1, 0].set_title(f"FNO prediction (modes={modes_x}x{modes_y})")
    fig.colorbar(im2, ax=axes[1, 0])
    
    err = np.abs(u_true - u_pred)
    im3 = axes[1, 1].imshow(err, origin='lower', cmap='inferno')
    axes[1, 1].set_title("Absolute error |u - u_pred|")
    fig.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    out_dir = "experiments/results/figures"
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "darcy_fno_results.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")

    # Secondary plot for loss
    plt.figure()
    plt.plot(losses)
    plt.yscale('log')
    plt.title("Darcy FNO Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "darcy_fno_loss.png"), dpi=150)
    print(f"Saved loss plot to {os.path.join(out_dir, 'darcy_fno_loss.png')}")
    plt.close()

if __name__ == "__main__":
    main()
