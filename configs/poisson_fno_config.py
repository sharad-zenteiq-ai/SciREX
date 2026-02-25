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
Configuration for FNO experiments on Poisson equation (2D and 3D).
"""

class FNO2DConfig:
    # Model Architecture
    n_modes = (16, 16)
    hidden_channels = 64
    n_layers = 4
    in_channels = 1
    out_channels = 1

    # Training Parameters
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 32
    epochs = 501  # Total training epochs
    steps_per_epoch = 100  # Number of batches per epoch

    # LR Scheduler (StepLR equivalent)
    scheduler_step_size = 100  # Decay LR every N epochs
    scheduler_gamma = 0.5  # Decay factor

    # Data Generation
    resolution = (64, 64)
    seed = 42


class FNO3DConfig:
    # Model Architecture
    n_modes = (12, 12, 12)
    hidden_channels = 32
    n_layers = 4
    in_channels = 4  # (source f, x, y, z)
    out_channels = 1

    # Training Parameters
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 4  # Reduced for 3D memory constraints
    epochs = 101    # Standard run length
    steps_per_epoch = 50 

    # LR Scheduler (StepLR equivalent)
    scheduler_step_size = 30
    scheduler_gamma = 0.5

    # Data Generation
    resolution = (32, 32, 32)
    include_mesh = True
    seed = 42
