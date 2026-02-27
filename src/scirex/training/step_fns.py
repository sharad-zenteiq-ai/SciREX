from typing import Callable, Dict, Tuple
import jax
from flax.training.train_state import TrainState

@jax.jit(static_argnames=("loss_fn",))
def train_step(state: TrainState, batch: Dict, loss_fn: Callable) -> Tuple[TrainState, Dict]:
    """
    Single train step.

    batch: dict with keys 'x' and 'y' (both jnp arrays)
    loss_fn: callable(preds, targets) -> scalar
    """
    def loss_and_preds(params):
        preds = state.apply_fn({"params": params}, batch["x"])
        loss = loss_fn(preds, batch["y"])
        return loss, preds

    (loss, preds), grads = jax.value_and_grad(loss_and_preds, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss}
    return state, metrics


@jax.jit(static_argnames=("loss_fn",))
def eval_step(state: TrainState, batch: Dict, loss_fn: Callable) -> Dict:
    """
    Forward evaluation step.
    """
    preds = state.apply_fn({"params": state.params}, batch["x"])
    loss = loss_fn(preds, batch["y"])
    return {"loss": loss, "preds": preds}
