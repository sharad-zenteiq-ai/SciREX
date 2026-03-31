"""
JAX/Flax/Optax conversion of the neuralop Trainer class.

Key paradigm differences from PyTorch:
  - Model parameters are explicit pytrees (dicts), not hidden inside nn.Module.
  - Training step is a pure function decorated with @jax.jit.
  - Gradients are computed with jax.value_and_grad instead of loss.backward().
  - Optimizer state is managed by optax and is explicit state.
  - No .to(device): JAX handles device placement via jax.device_put or automatically.
  - Mixed precision is handled by jax.lax.Precision or policy-based casting.

Dependencies:
    pip install jax flax optax
"""

from timeit import default_timer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from functools import partial
import sys

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from flax import struct
import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Params = Any          # flax parameter pytree
Batch  = Dict[str, Any]


# ---------------------------------------------------------------------------
# TrainState
# Extends flax's built-in TrainState to carry an optional regularizer state
# and a step counter that maps to "epoch" in the original code.
# ---------------------------------------------------------------------------
class TrainState(train_state.TrainState):
    """Extended TrainState that adds an optional regularizer state."""
    regularizer_state: Optional[Any] = None


# ---------------------------------------------------------------------------
# Helper: default Lp loss (matches neuralop.losses.LpLoss d=2 behaviour)
# Sums over the batch dimension (not mean), matching the original Trainer's
# expectation.
# ---------------------------------------------------------------------------
def lp_loss(pred: jnp.ndarray, y: jnp.ndarray, p: int = 2) -> jnp.ndarray:
    """Relative Lp loss, summed over the batch dimension."""
    diff_norm = jnp.linalg.norm(
        (pred - y).reshape(pred.shape[0], -1), ord=p, axis=-1
    )
    y_norm = jnp.linalg.norm(y.reshape(y.shape[0], -1), ord=p, axis=-1)
    return jnp.sum(diff_norm / (y_norm + 1e-8))


# ---------------------------------------------------------------------------
# Core training step (pure function, JIT-compiled)
# ---------------------------------------------------------------------------
def make_train_step(
    loss_fn: Callable,
    regularizer_fn: Optional[Callable] = None,
):
    """
    Factory that returns a JIT-compiled train_step function.

    Parameters
    ----------
    loss_fn : callable(pred, batch) -> scalar
        Training loss. Must sum (not mean) over the batch dim.
    regularizer_fn : callable(params) -> scalar, optional
        Extra penalty on parameters (e.g. Lasso).

    Returns
    -------
    train_step : callable
    """

    @jax.jit
    def train_step(state: TrainState, batch: Batch):
        """One gradient step.

        Parameters
        ----------
        state : TrainState
        batch : dict  {"x": ..., "y": ...}

        Returns
        -------
        new_state : TrainState
        loss_val : float
        reg_loss_val : float | None
        """

        def compute_loss(params):
            pred = state.apply_fn({"params": params}, batch["x"])
            loss = loss_fn(pred, batch["y"])
            reg = (
                regularizer_fn(params) if regularizer_fn is not None else 0.0
            )
            return loss + reg, (loss, reg)

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (total_loss, (loss_val, reg_val)), grads = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)
        return new_state, loss_val, reg_val

    return train_step


# ---------------------------------------------------------------------------
# Core eval step (pure function, JIT-compiled)
# ---------------------------------------------------------------------------
def make_eval_step():
    """Returns a JIT-compiled eval_step.

    loss_fns is passed as a tuple of (name, fn) pairs rather than a dict
    because dicts are not hashable and can't be used as static JIT args.
    The caller converts self.eval_losses.items() to a tuple before passing.
    """

    # loss_fns is a tuple of (name, callable) pairs — not JAX arrays.
    # static_argnames tells JIT to treat it as a compile-time constant.
    @partial(jax.jit, static_argnames=("loss_fns",))
    def eval_step(
        state: TrainState,
        batch: Batch,
        loss_fns: tuple,  # tuple of (name, fn) pairs
    ):
        """Run inference and compute all eval losses.

        Parameters
        ----------
        state : TrainState
        batch : dict  {"x": ..., "y": ...}
        loss_fns : tuple of (name, callable) pairs

        Returns
        -------
        step_losses : dict  name -> scalar
        pred : jnp.ndarray
        """
        pred = state.apply_fn({"params": state.params}, batch["x"])
        step_losses = {
            name: fn(pred, batch["y"]) for name, fn in loss_fns
        }
        return step_losses, pred

    return eval_step


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """
    JAX/Flax/Optax trainer for neural operators.

    Mirrors the PyTorch neuralop Trainer interface, adapted for JAX idioms.

    Parameters
    ----------
    model : flax.linen.Module
        The neural operator model.
    n_epochs : int
    optimizer : optax.GradientTransformation
        e.g. optax.adam(1e-3)
    scheduler : optax.Schedule, optional
        Learning-rate schedule. If provided, wrap your optimizer with
        ``optax.inject_hyperparams`` before passing it in, or pass the
        schedule directly; Trainer will inject it for you.
    training_loss : callable(pred, y) -> scalar, optional
        Defaults to relative L2 loss summed over batch.
    eval_losses : dict[str, callable], optional
        Keyed loss name -> loss function. Defaults to {"l2": training_loss}.
    regularizer : callable(params) -> scalar, optional
        Parameter regularizer.
    data_processor : object, optional
        Must implement .preprocess(batch) and .postprocess(pred, batch).
    eval_interval : int, default 1
    verbose : bool, default False
    save_dir : str | Path, default "./ckpt"
    save_best : str, optional
        Metric key to monitor for best-model checkpointing.
    save_every : int, optional
        Save checkpoint every N epochs.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        optimizer: optax.GradientTransformation,
        scheduler: Optional[optax.Schedule] = None,
        training_loss: Optional[Callable] = None,
        eval_losses: Optional[Dict[str, Callable]] = None,
        regularizer: Optional[Callable] = None,
        data_processor=None,
        eval_interval: int = 1,
        verbose: bool = False,
        save_dir: Union[str, Path] = "./ckpt",
        save_best: Optional[str] = None,
        save_every: Optional[int] = None,
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.eval_interval = eval_interval
        self.verbose = verbose
        self.data_processor = data_processor
        self.save_dir = Path(save_dir)
        self.save_best = save_best
        self.save_every = save_every

        # Losses
        self.training_loss = training_loss if training_loss is not None else lp_loss
        self.eval_losses = (
            eval_losses if eval_losses is not None else {"l2": self.training_loss}
        )
        self.regularizer = regularizer

        # Optimizer: optionally wrap with lr schedule
        if scheduler is not None:
            self.optimizer = optax.chain(
                optimizer,
                optax.scale_by_schedule(scheduler),
            )
        else:
            self.optimizer = optimizer

        # Build JIT-compiled steps
        self._train_step = make_train_step(
            loss_fn=self.training_loss,
            regularizer_fn=self.regularizer,
        )
        self._eval_step = make_eval_step()

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    def _init_state(self, sample_batch: Batch) -> TrainState:
        """Initialise model parameters and optimizer state.

        Parameters
        ----------
        sample_batch : dict
            A single batch used to trace the model's input shape.

        Returns
        -------
        state : TrainState
        """
        rng = jax.random.PRNGKey(0)
        x_sample = jnp.ones_like(jnp.array(sample_batch["x"][:1]))
        params = self.model.init(rng, x_sample)["params"]
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
        )
        return state

    # ------------------------------------------------------------------
    # Checkpoint helpers (simple numpy save / load via orbax or np.savez)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, state: TrainState, epoch: int, tag: str = "model"):
        """Save params and opt_state to disk using numpy serialisation."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{tag}_state.npz"
        # Flatten pytree to numpy for portability
        flat_params, treedef = jax.tree_util.tree_flatten(state.params)
        np.savez(
            path,
            epoch=np.array(epoch),
            **{f"p_{i}": np.array(v) for i, v in enumerate(flat_params)},
        )
        if self.verbose:
            print(f"Checkpoint saved → {path}")

    def _load_checkpoint(
        self, state: TrainState, tag: str = "model"
    ) -> tuple[TrainState, int]:
        """Load params from a checkpoint. Returns (updated_state, epoch)."""
        path = self.save_dir / f"{tag}_state.npz"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")
        data = np.load(path)
        epoch = int(data["epoch"])
        flat_params, treedef = jax.tree_util.tree_flatten(state.params)
        loaded = [jnp.array(data[f"p_{i}"]) for i in range(len(flat_params))]
        new_params = treedef.unflatten(loaded)
        state = state.replace(params=new_params)
        if self.verbose:
            print(f"Resumed from epoch {epoch} ← {path}")
        return state, epoch

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------
    def train(
        self,
        train_loader,
        test_loaders: Dict[str, Any],
        resume_from_dir: Optional[Union[str, Path]] = None,
        eval_modes: Optional[Dict[str, str]] = None,
        max_autoregressive_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train the model.

        Parameters
        ----------
        train_loader : iterable of dicts {"x": array, "y": array}
        test_loaders : dict  loader_name -> iterable
        resume_from_dir : str | Path, optional
            Resume training state from this directory.
        eval_modes : dict, optional
            Maps loader_name to "single_step" (default) or "autoregression".
        max_autoregressive_steps : int, optional
            Cap on autoregressive rollout steps.

        Returns
        -------
        epoch_metrics : dict
            Validation metrics from the final epoch.
        """
        if eval_modes is None:
            eval_modes = {}

        # Initialise state from first batch
        first_batch = next(iter(train_loader))
        state = self._init_state(first_batch)

        start_epoch = 0

        # Optionally resume
        if resume_from_dir is not None:
            self.save_dir = Path(resume_from_dir)
            tag = (
                "best_model"
                if (self.save_dir / "best_model_state.npz").exists()
                else "model"
            )
            state, start_epoch = self._load_checkpoint(state, tag=tag)

        if self.verbose:
            total_train = sum(1 for _ in train_loader)
            print(f"Training for {self.n_epochs} epochs.")

        best_metric_value = float("inf")
        epoch_metrics = {}

        for epoch in range(start_epoch, self.n_epochs):
            state, train_err, avg_loss, avg_reg_loss, epoch_time = (
                self._train_one_epoch_full(epoch, state, train_loader)
            )

            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_reg_loss=avg_reg_loss,
                epoch_time=epoch_time,
            )

            if epoch % self.eval_interval == 0:
                eval_metrics = self._evaluate_all(
                    epoch=epoch,
                    state=state,
                    test_loaders=test_loaders,
                    eval_modes=eval_modes,
                    max_autoregressive_steps=max_autoregressive_steps,
                )
                epoch_metrics.update(eval_metrics)

                if self.save_best is not None:
                    assert self.save_best in eval_metrics, (
                        f"save_best='{self.save_best}' not found in eval metrics: "
                        f"{list(eval_metrics.keys())}"
                    )
                    if eval_metrics[self.save_best] < best_metric_value:
                        best_metric_value = eval_metrics[self.save_best]
                        self._save_checkpoint(state, epoch, tag="best_model")

            if self.save_every is not None and self.save_best is None:
                if epoch % self.save_every == 0:
                    self._save_checkpoint(state, epoch, tag="model")

        return epoch_metrics

    # ------------------------------------------------------------------
    # One epoch (internal — returns new state + metrics)
    # ------------------------------------------------------------------
    def _train_one_epoch_full(self, epoch, state, train_loader):
        """Iterate over train_loader for one epoch.

        Returns
        -------
        state, train_err, avg_loss, avg_reg_loss, epoch_time
        """
        t0 = default_timer()
        total_loss = 0.0
        total_reg  = 0.0
        n_batches  = 0
        n_samples  = 0

        # Note: Flax has no .train()/.eval() toggle — dropout/BN are controlled
        # by passing `deterministic=True/False` in the model's apply() call instead.

        for batch in train_loader:
            batch = self._to_jax(batch)
            if self.data_processor is not None:
                batch = self.data_processor.preprocess(batch)

            state, loss_val, reg_val = self._train_step(state, batch)

            batch_size = batch["y"].shape[0]
            n_samples  += batch_size
            total_loss += float(loss_val)
            total_reg  += float(reg_val)
            n_batches  += 1

            if self.data_processor is not None:
                # postprocess is a no-op during training (labels already in batch)
                pass

        epoch_time = default_timer() - t0
        train_err  = total_loss / max(n_batches, 1)
        avg_loss   = total_loss / max(n_samples, 1)
        avg_reg    = total_reg  / max(n_samples, 1) if self.regularizer else None

        if self.verbose and epoch % self.eval_interval == 0:
            self._log_training(epoch, epoch_time, avg_loss, train_err, avg_reg)

        return state, train_err, avg_loss, avg_reg, epoch_time

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _evaluate_all(
        self,
        epoch: int,
        state: TrainState,
        test_loaders: Dict,
        eval_modes: Dict,
        max_autoregressive_steps: Optional[int],
    ) -> Dict[str, float]:
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            mode = eval_modes.get(loader_name, "single_step")
            metrics = self._evaluate(
                state=state,
                data_loader=loader,
                log_prefix=loader_name,
                mode=mode,
                max_steps=max_autoregressive_steps,
            )
            all_metrics.update(metrics)
        if self.verbose:
            self._log_eval(epoch, all_metrics)
        return all_metrics

    def _evaluate(
        self,
        state: TrainState,
        data_loader,
        log_prefix: str = "",
        mode: str = "single_step",
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate over an entire data_loader.

        Parameters
        ----------
        state : TrainState
        data_loader : iterable
        log_prefix : str
        mode : "single_step" | "autoregression"
        max_steps : int, optional

        Returns
        -------
        errors : dict  f"{log_prefix}_{loss_name}" -> float
        """
        errors = {f"{log_prefix}_{k}": 0.0 for k in self.eval_losses}
        n_samples = 0

        for idx, batch in enumerate(data_loader):
            batch = self._to_jax(batch)
            if self.data_processor is not None:
                batch = self.data_processor.preprocess(batch)

            if mode == "single_step":
                # Convert dict -> tuple of (name, fn) pairs: dicts aren't
                # hashable so can't be passed as static_argnames to jax.jit.
                loss_fns_tuple = tuple(self.eval_losses.items())
                step_losses, _ = self._eval_step(state, batch, loss_fns_tuple)
                n = batch["y"].shape[0]
                for k, v in step_losses.items():
                    errors[f"{log_prefix}_{k}"] += float(v)
                n_samples += n

            elif mode == "autoregression":
                step_losses, n = self._eval_autoregressive(
                    state, batch, max_steps=max_steps
                )
                for k, v in step_losses.items():
                    errors[f"{log_prefix}_{k}"] += float(v)
                n_samples += n

        for key in errors:
            errors[key] /= max(n_samples, 1)

        return errors

    def _eval_autoregressive(
        self,
        state: TrainState,
        batch: Batch,
        max_steps: Optional[int] = None,
    ):
        """Autoregressive rollout evaluation.

        At each step the model's output is fed back as the next input.
        Requires data_processor to implement preprocess(batch, step=t)
        and postprocess(pred, batch, step=t).

        Returns
        -------
        avg_losses : dict  loss_name -> average over steps
        n_samples : int
        """
        if max_steps is None:
            max_steps = float("inf")

        step_accum = {k: 0.0 for k in self.eval_losses}
        t = 0
        n_samples = None
        current_batch = batch

        while current_batch is not None and t < max_steps:
            if self.data_processor is not None:
                current_batch = self.data_processor.preprocess(current_batch, step=t)
            if current_batch is None:
                break

            if n_samples is None:
                n_samples = current_batch["y"].shape[0]

            pred = state.apply_fn({"params": state.params}, current_batch["x"])

            if self.data_processor is not None:
                pred, current_batch = self.data_processor.postprocess(
                    pred, current_batch, step=t
                )

            for k, loss_fn in self.eval_losses.items():
                step_accum[k] += float(loss_fn(pred, current_batch["y"]))

            t += 1

        # Average across steps
        if t > 0:
            step_accum = {k: v / t for k, v in step_accum.items()}

        return step_accum, n_samples or 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _to_jax(batch: dict) -> dict:
        """Convert a batch of numpy / torch tensors to JAX arrays."""
        out = {}
        for k, v in batch.items():
            if hasattr(v, "numpy"):          # torch.Tensor
                out[k] = jnp.array(v.numpy())
            elif isinstance(v, np.ndarray):
                out[k] = jnp.array(v)
            else:
                out[k] = v                   # already jnp or non-array
        return out

    def _log_training(self, epoch, time, avg_loss, train_err, avg_reg=None):
        msg = f"[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}"
        if avg_reg is not None:
            msg += f", avg_reg={avg_reg:.4f}"
        print(msg)
        sys.stdout.flush()

    def _log_eval(self, epoch, metrics):
        parts = [f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (float, int))]
        print(f"Eval [{epoch}]: " + ", ".join(parts))
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Minimal smoke-test using a tiny MLP as a stand-in for a neural operator.
    Replace with your actual Flax model and dataloaders.
    """

    # --- Toy Flax model ---
    class TinyMLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(64)(x)
            x = nn.relu(x)
            x = nn.Dense(32)(x)
            return x

    # --- Toy dataloader (list of batches) ---
    key = jax.random.PRNGKey(42)
    def make_loader(n_batches=8, batch_size=16, in_dim=32, out_dim=32):
        batches = []
        for _ in range(n_batches):
            x = np.random.randn(batch_size, in_dim).astype(np.float32)
            y = np.random.randn(batch_size, out_dim).astype(np.float32)
            batches.append({"x": x, "y": y})
        return batches

    train_loader = make_loader()
    test_loaders = {"val": make_loader(n_batches=4)}

    # --- Build trainer ---
    trainer = Trainer(
        model=TinyMLP(),
        n_epochs=5,
        optimizer=optax.adam(1e-3),
        verbose=True,
        eval_interval=1,
    )

    metrics = trainer.train(train_loader, test_loaders)
    print("Final metrics:", metrics)
