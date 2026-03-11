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
SciREX Model Parallel Utilities (MPU)

Provides tensor-parallel communication primitives for JAX-based
scientific machine learning models.

Supports scatter, gather, and reduction across model-parallel devices.
"""

from .comm import (
    init,
    get_world_size,
    get_rank,
    get_devices,
    is_parallel,
    get_model_parallel_axis,
)

from .mappings import (
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
    gather_from_model_parallel_region,
)

__all__ = [
    "init",
    "get_world_size",
    "get_rank",
    "get_devices",
    "is_parallel",
    "get_model_parallel_axis",
    "copy_to_model_parallel_region",
    "reduce_from_model_parallel_region",
    "scatter_to_model_parallel_region",
    "gather_from_model_parallel_region",
]