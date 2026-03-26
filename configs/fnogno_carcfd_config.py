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
Experiment configuration for FNOGNO on Car-CFD dataset.
"""

from typing import List, Optional, Literal, Any
from zencfg import ConfigBase
from configs.models import FNOGNOConfig, FNOGNO_Small3d

class FNOGNOCarCFDOptConfig(ConfigBase):
    n_epochs: int = 100
    learning_rate: float = 1e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "cosine" # Literal["step", "cosine"]
    step_size: int = 10
    gamma: float = 0.5
    cosine_decay_epochs: int = 100


class FNOGNOCarCFDDatasetConfig(ConfigBase):
    data_root: str = "./scirex/operators/data/car_cfd_data"
    batch_size: int = 1
    n_train: int = 100
    n_test: int = 20
    query_res: List[int] = [32, 32, 32]
    download: bool = True
    neighbor_cache_dir: str = "./scirex/operators/data/neighbor_cache"


class Default(ConfigBase):
    """Full experiment config for Car-CFD + FNOGNO."""
    
    verbose: bool = True
    model: FNOGNOConfig = FNOGNO_Small3d()
    opt: FNOGNOCarCFDOptConfig = FNOGNOCarCFDOptConfig()
    data: FNOGNOCarCFDDatasetConfig = FNOGNOCarCFDDatasetConfig()

    # ── Paths ──
    checkpoint_dir: str = "experiments/checkpoints"
    results_dir: str = "experiments/results/car_cfd_fnogno"
    model_name: str = "car_cfd_fnogno_jax.pkl"
    seed: int = 42

    # ── Logging ──
    wandb_log: bool = False

    def get_steps_per_epoch(self) -> int:
        return max(self.data.n_train // self.data.batch_size, 1)


# Alias for backward compatibility
FNOGNOCarCFDConfig = Default
