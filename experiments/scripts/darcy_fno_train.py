"""Train a Flax FNO to learn the 2D Darcy solution operator (a -> u)."""
import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.fno.models.fno2d import FNO2D
from scirex.operators.layers import Lifting, Projection
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse
from scirex.data.datasets.darcy import generator as darcy_generator

def main():
    # Model Config
    batch_size = 16
    nx, ny, in_ch = 64, 64, 1
    width = 64
    depth = 4
    modes_x = 16  # Number of Fourier modes to keep
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

    # Dataset config - Using FNO-style binary permeability {4, 12}
    num_batches = 500
    gen = darcy_generator(
        num_batches=num_batches, 
        batch_size=batch_size, 
        nx=nx, ny=ny, 
        rng_seed=0,
        mode="binary",  # FNO-style thresholded GRF
        a_low=4.0,
        a_high=12.0
    )

    print(f"Starting training (FNO 2D Darcy, modes={modes_x}x{modes_y})...")
    for step, (a_np, u_np) in enumerate(gen):
        batch = {"x": jnp.asarray(a_np), "y": jnp.asarray(u_np)}
        state, metrics = train_step(state, batch, mse)
        
        if step % 10 == 0:
            print(f"step {step:4d}, loss: {float(metrics['loss']):.6e}")
            
        if step == num_batches - 1 or (step > 0 and step % 50 == 0):
            eval_metrics = eval_step(state, batch, mse)
            print(f"  eval loss at step {step}: {float(eval_metrics['loss']):.6e}")

    # Final validation on fresh data
    print("Final evaluation...")
    a_test, u_test = next(darcy_generator(
        num_batches=1, batch_size=batch_size, nx=nx, ny=ny, 
        rng_seed=999, mode="binary", a_low=4.0, a_high=12.0
    ))
    eval_batch = {"x": jnp.asarray(a_test), "y": jnp.asarray(u_test)}
    final_metrics = eval_step(state, eval_batch, mse)
    print("Final eval loss:", float(final_metrics["loss"]))

if __name__ == "__main__":
    main()
