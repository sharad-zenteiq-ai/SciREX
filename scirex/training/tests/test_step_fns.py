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

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.models.fno import FNO2D
from scirex.training.train_state import create_train_state
from scirex.training.step_fns import train_step, eval_step


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def model_and_state(rng):
    """Create a small FNO2D model and its TrainState."""
    model = FNO2D(
        hidden_channels=8,
        n_layers=2,
        n_modes=(4, 4),
        out_channels=1,
    )
    input_shape = (2, 16, 16, 1)
    state = create_train_state(
        rng, model, input_shape, learning_rate=1e-3
    )
    return model, state


@pytest.fixture
def dummy_batch(rng):
    """Create a simple batch dict with 'x' and 'y' keys."""
    x = jax.random.normal(rng, (2, 16, 16, 1))
    y = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    return {"x": x, "y": y}


def mse_loss(preds, targets):
    """Simple MSE loss for testing."""
    return jnp.mean((preds - targets) ** 2)


# ─── train_step tests ─────────────────────────────────────────────────────────

class TestTrainStep:
    """Tests for the train_step function."""

    def test_returns_state_and_metrics(self, model_and_state, dummy_batch):
        """train_step should return (new_state, metrics_dict)."""
        _, state = model_and_state
        new_state, metrics = train_step(state, dummy_batch, mse_loss)
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_loss_is_scalar(self, model_and_state, dummy_batch):
        """The returned loss should be a scalar."""
        _, state = model_and_state
        _, metrics = train_step(state, dummy_batch, mse_loss)
        assert metrics["loss"].shape == (), f"Loss should be scalar, got shape {metrics['loss'].shape}"

    def test_loss_is_finite(self, model_and_state, dummy_batch):
        """The returned loss should be a finite number."""
        _, state = model_and_state
        _, metrics = train_step(state, dummy_batch, mse_loss)
        assert jnp.isfinite(metrics["loss"]), "Loss is not finite"

    def test_loss_is_non_negative(self, model_and_state, dummy_batch):
        """MSE loss should always be non-negative."""
        _, state = model_and_state
        _, metrics = train_step(state, dummy_batch, mse_loss)
        assert metrics["loss"] >= 0.0

    def test_step_counter_increments(self, model_and_state, dummy_batch):
        """state.step should increase by 1 after each train_step."""
        _, state = model_and_state
        assert state.step == 0
        new_state, _ = train_step(state, dummy_batch, mse_loss)
        assert new_state.step == 1

    def test_params_change_after_step(self, model_and_state, dummy_batch):
        """Parameters should be updated after a training step."""
        _, state = model_and_state
        new_state, _ = train_step(state, dummy_batch, mse_loss)

        # At least one parameter leaf should have changed
        old_leaves = jax.tree_util.tree_leaves(state.params)
        new_leaves = jax.tree_util.tree_leaves(new_state.params)

        any_changed = any(
            not jnp.allclose(old, new)
            for old, new in zip(old_leaves, new_leaves)
        )
        assert any_changed, "Parameters should change after a training step"

    def test_params_remain_finite(self, model_and_state, dummy_batch):
        """After a training step, all parameters should still be finite."""
        _, state = model_and_state
        new_state, _ = train_step(state, dummy_batch, mse_loss)
        for leaf in jax.tree_util.tree_leaves(new_state.params):
            assert jnp.all(jnp.isfinite(leaf)), "Non-finite parameter after train step"

    def test_multiple_steps_reduce_loss(self, model_and_state):
        """Training on the same batch repeatedly should reduce the loss."""
        _, state = model_and_state
        rng = jax.random.PRNGKey(99)
        x = jax.random.normal(rng, (2, 16, 16, 1))
        # Target is a simple scaled version of model output for easy fitting
        y = jnp.zeros_like(x)
        batch = {"x": x, "y": y}

        _, metrics_first = train_step(state, batch, mse_loss)
        initial_loss = metrics_first["loss"]

        # Run 50 steps
        for _ in range(50):
            state, _ = train_step(state, batch, mse_loss)

        _, metrics_last = train_step(state, batch, mse_loss)
        final_loss = metrics_last["loss"]

        assert final_loss < initial_loss, (
            f"Loss should decrease after training: initial={initial_loss:.6f}, "
            f"final={final_loss:.6f}"
        )


# ─── eval_step tests ──────────────────────────────────────────────────────────

class TestEvalStep:
    """Tests for the eval_step function."""

    def test_returns_dict_with_loss_and_preds(self, model_and_state, dummy_batch):
        """eval_step should return a dict with 'loss' and 'preds'."""
        _, state = model_and_state
        result = eval_step(state, dummy_batch, mse_loss)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "preds" in result

    def test_loss_is_scalar(self, model_and_state, dummy_batch):
        """The returned loss should be a scalar."""
        _, state = model_and_state
        result = eval_step(state, dummy_batch, mse_loss)
        assert result["loss"].shape == ()

    def test_loss_is_finite(self, model_and_state, dummy_batch):
        """The returned loss should be finite."""
        _, state = model_and_state
        result = eval_step(state, dummy_batch, mse_loss)
        assert jnp.isfinite(result["loss"])

    def test_preds_shape_matches_target(self, model_and_state, dummy_batch):
        """Predictions should have the same shape as the target."""
        _, state = model_and_state
        result = eval_step(state, dummy_batch, mse_loss)
        assert result["preds"].shape == dummy_batch["y"].shape, (
            f"Preds shape {result['preds'].shape} != target shape {dummy_batch['y'].shape}"
        )

    def test_preds_are_finite(self, model_and_state, dummy_batch):
        """Predictions should be finite values."""
        _, state = model_and_state
        result = eval_step(state, dummy_batch, mse_loss)
        assert jnp.all(jnp.isfinite(result["preds"])), "Predictions contain non-finite values"

    def test_eval_does_not_modify_state(self, model_and_state, dummy_batch):
        """eval_step should NOT change the model parameters (inference only)."""
        _, state = model_and_state
        _ = eval_step(state, dummy_batch, mse_loss)

        # state should be unchanged (no apply_gradients is called inside eval_step)
        assert state.step == 0, "eval_step should not modify the step counter"

    def test_eval_is_deterministic(self, model_and_state, dummy_batch):
        """Two eval_step calls with the same state and batch should produce identical results."""
        _, state = model_and_state
        result1 = eval_step(state, dummy_batch, mse_loss)
        result2 = eval_step(state, dummy_batch, mse_loss)
        assert jnp.allclose(result1["loss"], result2["loss"])
        assert jnp.allclose(result1["preds"], result2["preds"])

    def test_eval_loss_matches_manual(self, model_and_state, dummy_batch):
        """eval_step loss should match manually computed MSE."""
        _, state = model_and_state
        result = eval_step(state, dummy_batch, mse_loss)

        # Manual forward pass
        preds = state.apply_fn({"params": state.params}, dummy_batch["x"])
        expected_loss = jnp.mean((preds - dummy_batch["y"]) ** 2)

        assert jnp.allclose(result["loss"], expected_loss, atol=1e-6)
