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
Experiment configurations for FNO on Poisson equation (2D and 3D).

These configs compose a *model preset* from ``configs.models`` with
experiment-specific training and data-generation parameters.

Usage
-----
    from configs.poisson_fno_config import FNO2DConfig, FNO3DConfig

    config = FNO2DConfig()
    # config.model   → FNO_Medium2D instance  (architecture params)
    # config.*       → training / data params  (lr, batch_size, …)
"""

from dataclasses import dataclass, field
from typing import Literal
from configs.models import FNO_Medium2D, FNO_Medium3D, FNO_Large2D


@dataclass
class FNO2DConfig:
    """Full experiment config for 2D Poisson + FNO."""

    # ── Model Architecture (preset, can be swapped) ──
    model: FNO_Large2D = field(default_factory=FNO_Large2D)

    # ── Training Parameters ──
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 20
    epochs: int = 1000
    steps_per_epoch: int = 50
    n_test: int = 200

    # ── LR Scheduler ──
    scheduler_type: Literal["step", "cosine"] = "cosine"
    scheduler_step_size: int = 100   # used only when scheduler_type="step"
    scheduler_gamma: float = 0.5     # used only when scheduler_type="step"

    # ── Data Generation ──
    resolution: tuple = (64, 64)
    seed: int = 42

    # ── Convenience properties that proxy into the model preset ──
    @property
    def n_modes(self):
        return self.model.n_modes

    @property
    def hidden_channels(self):
        return self.model.hidden_channels

    @property
    def n_layers(self):
        return self.model.n_layers

    @property
    def in_channels(self):
        return self.model.in_channels

    @property
    def out_channels(self):
        return self.model.out_channels
    
    @property
    def lifting_channel_ratio(self):
        return self.model.lifting_channel_ratio

    @property
    def projection_channel_ratio(self):
        return self.model.projection_channel_ratio
    
    @property
    def use_grid(self):
        return self.model.use_grid

    @property
    def fno_skip(self):
        return self.model.fno_skip
    
    @property
    def use_channel_mlp(self):
        return self.model.use_channel_mlp

    @property
    def channel_mlp_skip(self):
        return self.model.channel_mlp_skip

    @property
    def domain_padding(self):
        return self.model.domain_padding


@dataclass
class FNO3DConfig:
    """Full experiment config for 3D Poisson + FNO."""

    # ── Model Architecture (preset, can be swapped) ──
    model: FNO_Medium3D = field(default_factory=FNO_Medium3D)

    # ── Training Parameters ──
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 101
    steps_per_epoch: int = 50

    # ── LR Scheduler ──
    scheduler_type: Literal["step", "cosine"] = "cosine"
    scheduler_step_size: int = 30    # used only when scheduler_type="step"
    scheduler_gamma: float = 0.5     # used only when scheduler_type="step"

    # ── Data Generation ──
    resolution: tuple = (32, 32, 32)
    include_mesh: bool = False # Now redundant as model handles it
    seed: int = 42

    # ── Convenience properties that proxy into the model preset ──
    @property
    def n_modes(self):
        return self.model.n_modes

    @property
    def hidden_channels(self):
        return self.model.hidden_channels

    @property
    def n_layers(self):
        return self.model.n_layers

    @property
    def in_channels(self):
        return self.model.in_channels

    @property
    def out_channels(self):
        return self.model.out_channels

    @property
    def lifting_channel_ratio(self):
        return self.model.lifting_channel_ratio

    @property
    def projection_channel_ratio(self):
        return self.model.projection_channel_ratio

    @property
    def use_grid(self):
        return self.model.use_grid

    @property
    def use_norm(self):
        return self.model.use_norm

    @property
    def fno_skip(self):
        return self.model.fno_skip
    
    @property
    def use_channel_mlp(self):
        return self.model.use_channel_mlp

    @property
    def channel_mlp_skip(self):
        return self.model.channel_mlp_skip

    @property
    def domain_padding(self):
        return self.model.domain_padding
