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

MODEL_PARALLEL_AXIS = "model_parallel"


def get_world_size():
    """
    Return number of available devices.
    """
    return jax.device_count()


def get_rank():
    """
    Return process index.
    """
    return jax.process_index()


def get_devices():
    """
    Return list of devices.
    """
    return jax.devices()


def is_parallel():
    """
    Return True if multiple devices are available.
    """
    return jax.device_count() > 1


def get_model_parallel_axis():
    """
    Axis name used for model parallel operations.
    """
    return MODEL_PARALLEL_AXIS


def init(verbose=True):
    """
    Initialize the parallel environment.
    """

    world_size = jax.device_count()

    if verbose:
        print("Initializing SciREX Model Parallel Utilities")
        print(f"Device count: {world_size}")
        print(f"Devices: {jax.devices()}")
        print(f"Model parallel axis: {MODEL_PARALLEL_AXIS}")

    return world_size