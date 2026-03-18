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
Experiment configuration for GINO on Car-CFD dataset.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List
from configs.models import GINO_Small3d


@dataclass
class GINOCarCFDConfig:
    """Full experiment config for Car-CFD + GINO."""

    # ── Model Architecture (preset) ──
    model: GINO_Small3d = field(default_factory=GINO_Small3d)

    # ── Training Parameters ──
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 20
    seed: int = 42

    # ── LR Scheduler ──
    scheduler_type: Literal["step", "cosine"] = "cosine"
    cosine_decay_epochs: int = 20
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5

    # ── Data Parameters ──
    data_root: str = "./scirex/operators/data/car_cfd_data"
    query_res: list = field(default_factory=lambda: [32, 32, 32])
    n_train: int = 500
    n_test: int = 20
    download: bool = True

    # ── Neighbor Search Cache ──
    neighbor_cache_dir: str = "./scirex/operators/data/neighbor_cache"

    # ── Logging ──
    wandb_log: bool = False

    # ── Convenience properties ──
    @property
    def fno_n_modes(self):
        return self.model.fno_n_modes

    @property
    def fno_hidden_channels(self):
        return self.model.fno_hidden_channels

    @property
    def fno_n_layers(self):
        return self.model.fno_n_layers

    @property
    def fno_lifting_channel_ratio(self):
        return self.model.fno_lifting_channel_ratio

    @property
    def in_gno_radius(self):
        return self.model.in_gno_radius

    @property
    def out_gno_radius(self):
        return self.model.out_gno_radius

    @property
    def in_channels(self):
        return self.model.in_channels

    @property
    def out_channels(self):
        return self.model.out_channels

    @property
    def gno_coord_dim(self):
        return self.model.gno_coord_dim

    @property
    def in_gno_transform_type(self):
        return self.model.in_gno_transform_type

    @property
    def out_gno_transform_type(self):
        return self.model.out_gno_transform_type

    @property
    def max_neighbors(self):
        return self.model.max_neighbors

    @property
    def use_neighbor_cache(self):
        return self.model.use_neighbor_cache