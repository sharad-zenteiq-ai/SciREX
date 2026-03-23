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

"""
Integration tests for GNOBlock.
"""

import sys
import os

# Resolve ModuleNotFoundError when running as a script from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.gno_block import GNOBlock

# Fixed variables
in_channels = 3
out_channels = 3
mlp_hidden_layers = [16, 16, 16]

# data parameters
n_in = 100
n_out = 100
radius = 0.25

RNG = jax.random.PRNGKey(42)


def _make_points(rng, n, d):
    return jax.random.uniform(rng, (n, d))


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("gno_coord_dim", [2, 3])
@pytest.mark.parametrize("gno_pos_embed_type", ["nerf", "transformer", None])
@pytest.mark.parametrize(
    "gno_transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"]
)
def test_gno_block(
    gno_transform_type,
    gno_coord_dim,
    gno_pos_embed_type,
    batch_size,
):
    rng1, rng2, rng3, rng_init = jax.random.split(RNG, 4)

    block = GNOBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        coord_dim=gno_coord_dim,
        radius=radius,
        transform_type=gno_transform_type,
        pos_embedding_type=gno_pos_embed_type,
        channel_mlp_layers=mlp_hidden_layers,
    )

    y = _make_points(rng1, n_in, gno_coord_dim)
    x = _make_points(rng2, n_out, gno_coord_dim)

    f_y = None
    if gno_transform_type != "linear":
        f_y = jax.random.normal(rng3, (batch_size, n_in, in_channels))

    params = block.init(rng_init, y, x, f_y=f_y)
    out = block.apply(params, y, x, f_y=f_y)

    # Check output size
    # Batched outputs only matter in the nonlinear kernel use case
    if gno_transform_type != "linear":
        assert list(out.shape) == [batch_size, n_out, out_channels]
    else:
        assert list(out.shape) == [n_out, out_channels]

    # Check forward pass is finite
    assert jnp.all(jnp.isfinite(out))

    # Backward pass dummy loss calculation
    def loss_fn(p, y_in, x_in, f_y_in):
        pred = block.apply(p, y=y_in, x=x_in, f_y=f_y_in)
        if batch_size > 1 and gno_transform_type != "linear":
            return jnp.sum(pred[0])
        else:
            return jnp.sum(pred)

    # Check gradients with respect to f_y
    if f_y is not None:
        grad_fn = jax.grad(loss_fn, argnums=3)
        grad_f_y = grad_fn(params, y, x, f_y)
        
        # If batch size > 1, f_y[1:] gets zero gradient because loss is pred[0].sum()
        if batch_size > 1:
            assert jnp.all(grad_f_y[1:] == 0.0)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
