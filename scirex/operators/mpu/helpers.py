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

from .comm import get_world_size


def split_tensor_along_dim(x, dim, num_chunks):
    """
    Split tensor along a given dimension.
    """

    if x.shape[dim] % num_chunks != 0:
        raise ValueError(
            f"Cannot split dimension {dim} of size {x.shape[dim]} into {num_chunks} chunks"
        )

    return jnp.split(x, num_chunks, axis=dim)


def reduce(x, axis_name):
    """
    All-reduce tensor across devices.
    """

    if get_world_size() == 1:
        return x

    return jax.lax.psum(x, axis_name)


def gather(x, axis_name="model_parallel"):
    """
    Gather tensor across model parallel devices.
    """
    if jax.device_count() == 1:
        return x

    # all_gather stacks tensors along a new leading axis
    gathered = jax.lax.all_gather(x, axis_name)

    return gathered

def scatter(x, dim, axis_name):
    """
    Scatter tensor across devices.
    """

    num_devices = get_world_size()

    if num_devices == 1:
        return x

    if x.shape[dim] % num_devices != 0:
        raise ValueError(
            f"Tensor dimension {x.shape[dim]} must be divisible by number of devices {num_devices}"
        )

    chunks = jnp.split(x, num_devices, axis=dim)

    # stack chunks so JAX can index them
    stacked = jnp.stack(chunks)

    idx = jax.lax.axis_index(axis_name)

    return jax.lax.dynamic_index_in_dim(stacked, idx, keepdims=False)