"""Train a Flax FNO to learn the Poisson solution operator (f -> u)."""
import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.fno.models.fno2d import FNO2D
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step
from scirex.losses.data_losses import mse
from scirex.data.datasets.poisson import generator as poisson_generator

def main():
    batch_size = 32
    nx, ny, in_ch = 64, 64, 1
    width = 64
    depth = 4
    modes_x = 16
    modes_y = 16
    out_ch = 1
    lr = 1e-3
    rng = jax.random.PRNGKey(42)

    model = FNO2D(width=width, depth=depth, modes_x=modes_x, modes_y=modes_y, out_channels=out_ch)
    state = create_train_state(rng, model, (batch_size, nx, ny, in_ch), learning_rate=lr)

    num_batches = 10
    gen = poisson_generator(num_batches=num_batches, batch_size=batch_size, nx=nx, ny=ny, channels=in_ch, rng_seed=0)

    print("Starting training (FNO 2D Poisson)...")
    for step, (f_np, u_np) in enumerate(gen):
        batch = {"x": jnp.asarray(f_np), "y": jnp.asarray(u_np)}
        state, metrics = train_step(state, batch, mse)
        if step % 100 == 0:
            print(f"step {step:4d}, loss: {float(metrics['loss']):.6e}")
        if step == num_batches - 1 or step % 500 == 0:
            eval_metrics = eval_step(state, batch, mse)
            print(f"  eval loss: {float(eval_metrics['loss']):.6e}")

    f_test, u_test = next(poisson_generator(num_batches=1, batch_size=batch_size, nx=nx, ny=ny, channels=in_ch, rng_seed=999))
    eval_batch = {"x": jnp.asarray(f_test), "y": jnp.asarray(u_test)}
    final_metrics = eval_step(state, eval_batch, mse)
    print("Final eval loss:", float(final_metrics["loss"]))

if __name__ == "__main__":
    main()
