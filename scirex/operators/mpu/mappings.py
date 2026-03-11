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

import jax.numpy as jnp

from .helpers import reduce
from .helpers import gather
from .helpers import scatter
from .comm import get_model_parallel_axis


def copy_to_model_parallel_region(x):
    """
    Identity operation for compatibility.
    """
    return x


def reduce_from_model_parallel_region(x):
    """
    Reduce tensor across model-parallel devices.
    """

    axis = get_model_parallel_axis()

    return reduce(x, axis)


def scatter_to_model_parallel_region(x, dim=-1):
    """
    Split tensor across model-parallel devices.
    """

    axis = get_model_parallel_axis()

    return scatter(x, dim, axis)


def gather_from_model_parallel_region(x, dim=-1):
    """
    Gather tensor from model parallel devices.
    """
    axis = get_model_parallel_axis()

    gathered = gather(x, axis)

    # gathered shape: (num_devices, local_shape...)
    # move device axis to the split dimension
    
    gathered = jnp.moveaxis(gathered, 0, dim)

    # merge gathered dimension
    return jnp.concatenate(gathered, axis=dim)