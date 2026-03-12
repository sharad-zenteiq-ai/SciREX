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
Experiment configuration for FNO2D on the neuraloperator 1-D Burgers dataset.

Dataset (confirmed structure)
------------------------------
    burgers_train_16.pt  →  x:(1200, 16)  y:(1200, 16, 17)  visc:0.01
    burgers_test_16.pt   →  x:(1200, 16)  y:(1200, 16, 17)  visc:0.01

    'x' : initial condition  u₀(x)
    'y' : full trajectory    u(x, t),  17 time steps
    'visc' : ν = 0.01

Operator learned
-----------------
    u₀(x)  ──FNO2D──>  u(x, T_final)

FNO2D input shape after loading:  (N, 16, 1, 2)   [u₀, grid_x]
FNO2D target shape:               (N, 16, 1, 1)   u at last time step

Dataset paths
-------------
    Default paths match the SciREX repository layout:
        scirex/operators/data/burgers_train_16.pt
        scirex/operators/data/burgers_test_16.pt

    Override via environment variables at runtime:
        BURGERS_TRAIN_PATH=/your/path/burgers_train_16.pt  python scripts/...

Usage
-----
    from configs.burgers_pt_fno_config import BurgersPtFNO2DConfig
    config = BurgersPtFNO2DConfig()
"""

from dataclasses import dataclass, field
from typing import Literal
from configs.models import FNO_Medium2D


@dataclass
class BurgersPtFNO2DConfig:
    """Full experiment config: neuraloperator Burgers .pt  +  SciREX FNO2D."""

    # ── Dataset paths ────────────────────────────────────────────────────
    train_path: str = "scirex/operators/data/burgers_train_16.pt"
    test_path:  str = "scirex/operators/data/burgers_test_16.pt"

    # ── Dataset sizes (max available: 1200 train, 1200 test) ─────────────
    n_train: int = 1000
    n_test:  int = 200

    # ── Spatial resolution (nx=16, ny=1 trivial dim) ─────────────────────
    nx: int = 16      # spatial points along x
    ny: int = 1       # trivial second dimension for FNO2D

    # ── Model Architecture ───────────────────────────────────────────────
    # FNO_Medium2D preset: n_modes=(24,24), hidden=128, n_layers=4
    # n_modes are automatically capped at (nx//2, ny//2) = (8, 1) for
    # the actual grid size — see the n_modes property below.
    model: FNO_Medium2D = field(default_factory=FNO_Medium2D)

    # ── Training ─────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    batch_size:    int   = 20
    epochs:        int   = 50
    steps_per_epoch: int = 50      # overridden at runtime from n_train

    # ── Normalisation ─────────────────────────────────────────────────────
    encode_input:  bool = True
    encode_output: bool = True

    # ── LR Scheduler ─────────────────────────────────────────────────────
    scheduler_type:       Literal["step", "cosine"] = "cosine"
    cosine_decay_epochs:  int   = 500
    scheduler_step_size:  int   = 100
    scheduler_gamma:      float = 0.5

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42

    # ── Convenience properties (proxy to model preset) ────────────────────

    @property
    def n_modes(self):
        """
        Fourier modes capped at Nyquist for the actual grid.

        For nx=16, ny=1:
            modes_x = min(preset, nx//2) = min(24, 8) = 8
            modes_y = min(preset, max(ny//2,1)) = 1
        """
        mx = min(self.model.n_modes[0], self.nx // 2)
        my = min(self.model.n_modes[1], max(self.ny // 2, 1))
        return (mx, my)

    @property
    def hidden_channels(self):          return self.model.hidden_channels
    @property
    def n_layers(self):                 return self.model.n_layers
    @property
    def in_channels(self):              return self.model.in_channels
    @property
    def out_channels(self):             return self.model.out_channels
    @property
    def lifting_channel_ratio(self):    return self.model.lifting_channel_ratio
    @property
    def projection_channel_ratio(self): return self.model.projection_channel_ratio
    @property
    def use_grid(self):                 return self.model.use_grid
    @property
    def fno_skip(self):                 return self.model.fno_skip
    @property
    def channel_mlp_skip(self):         return self.model.channel_mlp_skip
    @property
    def use_channel_mlp(self):          return self.model.use_channel_mlp
    @property
    def use_norm(self):                 return self.model.use_norm
    @property
    def domain_padding(self):           return self.model.domain_padding
