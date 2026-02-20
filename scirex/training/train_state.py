from typing import Any
import jax
import optax
from flax.training import train_state

# Alias for type hinting
TrainState = train_state.TrainState

def create_train_state(rng: Any, model: Any, input_shape: tuple, learning_rate: float, weight_decay: float = 1e-4) -> TrainState:
    """
    Initialize model params and optimizer state.

    - rng: PRNG key
    - model: Flax Module instance (e.g., FNO2D(...))
    - input_shape: shape tuple for dummy input (batch, nx, ny, channels)
    - learning_rate: optimizer lr (can be float or optax schedule)
    - weight_decay: weight decay for AdamW
    """
    dummy = jax.random.normal(rng, input_shape)
    variables = model.init(rng, dummy)
    params = variables["params"]
    # Using AdamW which is standard for FNO training
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state
