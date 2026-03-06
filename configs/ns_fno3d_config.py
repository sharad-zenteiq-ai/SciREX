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
Experiment configuration for Navier-Stokes 2D using FNO3D.

Dataset: nsforcing from the neuraloperator Zenodo archive.
  - x: vorticity input (forcing), y: vorticity output (solution)
  - Each sample is a 2D field on a unit square grid.
  - A trivial z-dimension (size 1) is added to use FNO3D on 2D snapshots.

Reference: https://github.com/neuraloperator/neuraloperator

Usage
-----
    from configs.ns_fno3d_config import NSFNO3DConfig, NSFNO3D_Small

    # Use default model (Large)
    config = NSFNO3DConfig()
    
    # Or override with a medium model
    config = NSFNO3DConfig(model=NSFNO3D_Medium())
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple
from configs.models import SimpleFNOConfig


# ═══════════════════════════════════════════════════════════════════
#  NS FNO3D  P R E S E T S
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NSFNO3D_Small(SimpleFNOConfig):
    """Small FNO3D architecture preset tuned for 2D Navier-Stokes."""
    n_modes: Tuple[int, int, int] = (16, 16, 1)
    hidden_channels: int = 32
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1
    use_grid: bool = True
    use_norm: bool = False


@dataclass
class NSFNO3D_Medium(SimpleFNOConfig):
    """Medium FNO3D architecture preset tuned for 2D Navier-Stokes.
    Matches the original reference script baseline."""
    n_modes: Tuple[int, int, int] = (32, 32, 1)
    hidden_channels: int = 64
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1
    use_grid: bool = True
    use_norm: bool = False


@dataclass
class NSFNO3D_Large(SimpleFNOConfig):
    """Large FNO3D architecture preset tuned for 2D Navier-Stokes.
    Increases hidden channels for more representational power."""
    n_modes: Tuple[int, int, int] = (32, 32, 1)
    hidden_channels: int = 128
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1
    use_grid: bool = True
    use_norm: bool = False


@dataclass
class NSFNO3D_Huge(SimpleFNOConfig):
    """Huge FNO3D architecture preset tuned for 2D Navier-Stokes.
    Maximum complexity for single-GPU training."""
    n_modes: Tuple[int, int, int] = (32, 32, 1)
    hidden_channels: int = 256
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1
    use_grid: bool = True
    use_norm: bool = False


# ═══════════════════════════════════════════════════════════════════
#  E X P E R I M E N T   C O N F I G
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NSFNO3DConfig:
    """Full experiment config for Navier-Stokes 2D + FNO3D."""

    # ── Model Architecture (preset, can be swapped) ──
    # Default to Medium to match neuraloperator reference
    model: SimpleFNOConfig = field(default_factory=NSFNO3D_Medium)

    # ── Dataset Paths ──
    data_dir: str = "/media/HDD/mamta_backup/datasets/fno/navier_stokes"
    train_file: str = "ns_train_64.pt"
    test_file: str = "ns_test_64.pt"

    # ── Data ──
    # Note: ns_train_64.pt only contains 1000 samples. 
    # n_train = 1000 is used even if 10000 is requested.
    n_train: int = 1000
    n_test: int = 200
    encode_input: bool = True
    encode_output: bool = True

    # ── Training Parameters ──
    # Ref: neuraloperator default 3e-4, 600 epochs
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 100
    seed: int = 42

    # ── LR Scheduler ──
    scheduler_type: Literal["step", "cosine"] = "step"
    scheduler_step_size: int = 100  # Decay LR every N epochs
    scheduler_gamma: float = 0.5
    cosine_decay_epochs: int = 600

    # ── Loss ──
    train_loss: str = "h1"  # "h1" or "lp"

    # ── Checkpoint / Results ──
    checkpoint_dir: str = "experiments/checkpoints"
    results_dir: str = "experiments/results/ns_fno3d"
    model_name: str = "ns_fno3d_jax.pkl"

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
    def use_norm(self):
        return self.model.use_norm

    @property
    def domain_padding(self):
        return self.model.domain_padding

    @property
    def steps_per_epoch(self):
        return self.n_train // self.batch_size
