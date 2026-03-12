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
from flax.training.train_state import TrainState

from scirex.operators.models.fno import FNO
from scirex.operators.training.train_state import create_train_state


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def simple_model():
    """A minimal FNO model for testing."""
    return FNO(
        hidden_channels=8,
        n_layers=2,
        n_modes=(4, 4),
        out_channels=1,
    )


class TestCreateTrainState:
    """Tests for the create_train_state factory function."""

    def test_returns_train_state(self, rng, simple_model):
        """create_train_state should return a Flax TrainState object."""
        state = create_train_state(
            rng, simple_model, (1, 16, 16, 1), learning_rate=1e-3
        )
        assert isinstance(state, TrainState)

    def test_params_are_populated(self, rng, simple_model):
        """The returned state must contain non-empty parameter pytree."""
        state = create_train_state(
            rng, simple_model, (1, 16, 16, 1), learning_rate=1e-3
        )
        # params should be a non-empty dict-like structure
        flat_params = jax.tree_util.tree_leaves(state.params)
        assert len(flat_params) > 0, "Params pytree should not be empty"

    def test_params_are_finite(self, rng, simple_model):
        """All initialised parameters should be finite (no NaN/Inf)."""
        state = create_train_state(
            rng, simple_model, (1, 16, 16, 1), learning_rate=1e-3
        )
        for leaf in jax.tree_util.tree_leaves(state.params):
            assert jnp.all(jnp.isfinite(leaf)), "Found non-finite values in params"

    def test_apply_fn_is_set(self, rng, simple_model):
        """state.apply_fn should be the model's apply method."""
        state = create_train_state(
            rng, simple_model, (1, 16, 16, 1), learning_rate=1e-3
        )
        assert state.apply_fn is not None
        # Should be callable
        assert callable(state.apply_fn)

    def test_forward_pass_with_state(self, rng, simple_model):
        """The state's apply_fn + params should produce a valid forward pass."""
        input_shape = (2, 16, 16, 1)
        state = create_train_state(
            rng, simple_model, input_shape, learning_rate=1e-3
        )
        x = jax.random.normal(rng, input_shape)
        preds = state.apply_fn({"params": state.params}, x)
        assert preds.shape == (2, 16, 16, 1), f"Expected (2,16,16,1), got {preds.shape}"
        assert jnp.all(jnp.isfinite(preds)), "Forward pass produced non-finite output"

    def test_step_counter_starts_at_zero(self, rng, simple_model):
        """TrainState.step should initialise to 0."""
        state = create_train_state(
            rng, simple_model, (1, 16, 16, 1), learning_rate=1e-3
        )
        assert state.step == 0

    def test_different_learning_rates(self, rng, simple_model):
        """States with different LRs should both be valid (smoke test)."""
        for lr in [1e-2, 1e-3, 1e-5]:
            state = create_train_state(
                rng, simple_model, (1, 16, 16, 1), learning_rate=lr
            )
            assert isinstance(state, TrainState)

    def test_weight_decay_default(self, rng, simple_model):
        """Default weight_decay=1e-4 should produce a valid state."""
        state = create_train_state(
            rng, simple_model, (1, 16, 16, 1), learning_rate=1e-3
        )
        assert isinstance(state, TrainState)

    def test_weight_decay_zero(self, rng, simple_model):
        """weight_decay=0 should be equivalent to plain Adam."""
        state = create_train_state(
            rng, simple_model, (1, 16, 16, 1), learning_rate=1e-3, weight_decay=0.0
        )
        assert isinstance(state, TrainState)

    def test_different_input_shapes(self, rng, simple_model):
        """create_train_state should work with various spatial resolutions."""
        for res in [8, 16, 32]:
            state = create_train_state(
                rng, simple_model, (1, res, res, 1), learning_rate=1e-3
            )
            x = jax.random.normal(rng, (1, res, res, 1))
            preds = state.apply_fn({"params": state.params}, x)
            assert preds.shape == (1, res, res, 1)

    def test_apply_gradients_increments_step(self, rng, simple_model):
        """Calling apply_gradients should increment state.step."""
        input_shape = (1, 16, 16, 1)
        state = create_train_state(
            rng, simple_model, input_shape, learning_rate=1e-3
        )

        # Compute dummy gradients
        x = jax.random.normal(rng, input_shape)

        def loss_fn(params):
            preds = state.apply_fn({"params": params}, x)
            return jnp.mean(preds ** 2)

        grads = jax.grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grads)
        assert new_state.step == 1
