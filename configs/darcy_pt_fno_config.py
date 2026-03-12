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
Experiment configuration for FNO2D on the neuraloperator Darcy Flow dataset.

Dataset
-------
Original FNO benchmark .pt files:
    darcy_train_16.pt  and  darcy_test_16.pt
    (from https://github.com/neuraloperator/neuraloperator)

Keys in each file:
    'x' : (N, 16, 16)  permeability a(x,y)
    'y' : (N, 16, 16)  pressure     u(x,y)

Operator learned:   a(x,y)  ──FNO2D──>  u(x,y)

Usage
-----
    from configs.darcy_pt_fno_config import DarcyPtFNO2DConfig
    config = DarcyPtFNO2DConfig()
    # Override dataset paths as needed:
    # config.train_path = "/my/path/darcy_train_16.pt"
"""

from dataclasses import dataclass, field
from typing import Literal
from configs.models import FNO_Medium2D


@dataclass
class DarcyPtFNO2DConfig:
    """Full experiment config: neuraloperator Darcy .pt  +  SciREX FNO2D."""

    # ── Dataset paths (neuraloperator convention) ────────────────────────
    train_path: str = "scirex/operators/data/darcy_test_16.pt"
    test_path:  str = "scirex/operators/data/darcy_test_16.pt"
    resolution: int = 16          # spatial grid size inferred from data

    # ── Subset sizes ─────────────────────────────────────────────────────
    n_train: int = 1000
    n_test:  int = 200

    # ── Model Architecture ───────────────────────────────────────────────
    # FNO_Medium2D: n_modes=(24,24), hidden=128, n_layers=4, use_norm=True
    # For 16×16 grids modes are capped at 8 automatically in training script.
    model: FNO_Medium2D = field(default_factory=FNO_Medium2D)

    # ── Training ─────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    batch_size:    int   = 20
    epochs:        int   = 500
    steps_per_epoch: int = 50      # overridden at runtime from n_train

    # ── Normalisation ─────────────────────────────────────────────────────
    # Both input (permeability) and output (pressure) are normalised to
    # zero mean / unit variance before training and decoded for metrics.
    encode_input:  bool = True
    encode_output: bool = True

    # ── LR Scheduler ────────────────────────────────────────────────────
    scheduler_type:       Literal["step", "cosine"] = "cosine"
    cosine_decay_epochs:  int   = 500
    scheduler_step_size:  int   = 100
    scheduler_gamma:      float = 0.5

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42

    # ── Convenience properties (proxy to model preset) ────────────────────
    @property
    def n_modes(self):
        # Cap Fourier modes at resolution // 2  (Nyquist limit)
        mx = min(self.model.n_modes[0], self.resolution // 2)
        my = min(self.model.n_modes[1], self.resolution // 2)
        return (mx, my)

    @property
    def hidden_channels(self):       return self.model.hidden_channels
    @property
    def n_layers(self):              return self.model.n_layers
    @property
    def in_channels(self):           return self.model.in_channels
    @property
    def out_channels(self):          return self.model.out_channels
    @property
    def lifting_channel_ratio(self): return self.model.lifting_channel_ratio
    @property
    def projection_channel_ratio(self): return self.model.projection_channel_ratio
    @property
    def use_grid(self):              return self.model.use_grid
    @property
    def fno_skip(self):              return self.model.fno_skip
    @property
    def channel_mlp_skip(self):      return self.model.channel_mlp_skip
    @property
    def use_channel_mlp(self):       return self.model.use_channel_mlp
    @property
    def use_norm(self):              return self.model.use_norm
    @property
    def domain_padding(self):        return self.model.domain_padding
