from typing import Any
import jax
import optax
from flax.training import train_state

# Alias for type hinting
TrainState = train_state.TrainState

def create_train_state(rng: Any, model: Any, input_shape: tuple, learning_rate: float) -> TrainState:
    """
    Initialize model params and optimizer state.

    - rng: PRNG key
    - model: Flax Module instance (e.g., FNO2D(...))
    - input_shape: shape tuple for dummy input (batch, nx, ny, channels)
    - learning_rate: optimizer lr
    """
    dummy = jax.random.normal(rng, input_shape)
    variables = model.init(rng, dummy)
    params = variables["params"]
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state
