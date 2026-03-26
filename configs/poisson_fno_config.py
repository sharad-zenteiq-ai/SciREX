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
Experiment configuration for Poisson equation using FNO (2D and 3D).
"""

from typing import List, Optional, Any, Tuple
from zencfg import ConfigBase
from configs.models import FNOConfig, SimpleFNOConfig, FNO_Medium2D, FNO_Medium3D

class PoissonOptConfig(ConfigBase):
    n_epochs: int = 100
    learning_rate: float = 5e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    step_size: int = 100
    gamma: float = 0.5
    cosine_decay_epochs: int = 100


class PoissonDatasetConfig(ConfigBase):
    n_train: int = 1000
    n_test: int = 200
    batch_size: int = 32
    resolution: Tuple[int, ...] = (64, 64)
    encode_input: bool = True
    encode_output: bool = True


class Poisson2DDefault(ConfigBase):
    """Full experiment config for 2D Poisson + FNO2D."""
    
    verbose: bool = True
    model: SimpleFNOConfig = FNO_Medium2D()
    opt: PoissonOptConfig = PoissonOptConfig()
    data: PoissonDatasetConfig = PoissonDatasetConfig()

    # ── Paths ──
    checkpoint_dir: str = "experiments/checkpoints"
    results_dir: str = "experiments/results/poisson_fno2d"
    model_name: str = "poisson_fno2d_jax.pkl"
    seed: int = 42

    def get_steps_per_epoch(self) -> int:
        return max(self.data.n_train // self.data.batch_size, 1)


class Poisson3DDefault(ConfigBase):
    """Full experiment config for 3D Poisson + FNO3D."""
    
    verbose: bool = True
    model: SimpleFNOConfig = FNO_Medium3D()
    opt: PoissonOptConfig = PoissonOptConfig(learning_rate=1e-3)
    data: PoissonDatasetConfig = PoissonDatasetConfig(
        batch_size=10, 
        resolution=(32, 32, 32),
        n_test=20
    )

    # ── Paths ──
    checkpoint_dir: str = "experiments/checkpoints"
    results_dir: str = "experiments/results/poisson_fno3d"
    model_name: str = "poisson_fno3d_jax.pkl"
    seed: int = 42

    def get_steps_per_epoch(self) -> int:
        return max(self.data.n_train // self.data.batch_size, 1)


# Aliases for backward compatibility
FNO2DConfig = Poisson2DDefault
FNO3DConfig = Poisson3DDefault
Default = Poisson2DDefault # Default to 2D
