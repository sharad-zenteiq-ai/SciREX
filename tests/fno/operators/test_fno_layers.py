"""
Unit test for forward pass shape of the FNO2D model.
"""
import jax
import jax.numpy as jnp
from scirex.operators.fno.models.fno2d import FNO2D
from scirex.training.train_state import create_train_state

def test_forward_shape():
    rng = jax.random.PRNGKey(0)
    batch, nx, ny, in_ch = 2, 16, 16, 1
    width = 16
    depth = 2
    modes_x = 6
    modes_y = 6
    out_ch = 1

    model = FNO2D(width=width, depth=depth, modes_x=modes_x, modes_y=modes_y, out_channels=out_ch)
    state = create_train_state(rng, model, (batch, nx, ny, in_ch), learning_rate=1e-3)
    x = jnp.ones((batch, nx, ny, in_ch), dtype=jnp.float32)
    
    preds = state.apply_fn({"params": state.params}, x)
    assert preds.shape == (batch, nx, ny, out_ch)
    print("test_forward_shape OK:", preds.shape)

if __name__ == "__main__":
    test_forward_shape()
