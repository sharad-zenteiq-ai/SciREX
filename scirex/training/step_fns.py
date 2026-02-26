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
