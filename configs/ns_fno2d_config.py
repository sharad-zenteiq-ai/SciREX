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
Experiment configuration for Navier-Stokes 2D using FNO2D.
Aligned with neuraloperator (https://github.com/neuraloperator/neuraloperator).
"""

from typing import List, Optional, Any
from zencfg import ConfigBase
from configs.models import SimpleFNOConfig, FNO_Medium2D

class NavierStokesOptConfig(ConfigBase):
    n_epochs: int = 100
    learning_rate: float = 3e-4
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 100
    gamma: float = 0.5


class NavierStokesDatasetConfig(ConfigBase):
    folder: str = "/media/HDD/mamta_backup/datasets/fno/navier_stokes"
    train_file: str = "nsforcing_train_128.pt"
    test_file: str = "nsforcing_test_128.pt"
    batch_size: int = 8
    n_train: int = 1000
    train_resolution: int = 128
    n_tests: List[int] = [200]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
    encode_input: bool = True
    encode_output: bool = True


class Default(ConfigBase):
    """Full experiment config for Navier-Stokes 2D + FNO2D."""
    
    verbose: bool = True
    model: SimpleFNOConfig = FNO_Medium2D()
    opt: NavierStokesOptConfig = NavierStokesOptConfig()
    data: NavierStokesDatasetConfig = NavierStokesDatasetConfig()

    # ── Paths ──
    checkpoint_dir: str = "experiments/checkpoints"
    results_dir: str = "experiments/results/ns_fno2d"
    model_name: str = "ns_fno2d_jax.pkl"
    seed: int = 666

    def get_steps_per_epoch(self) -> int:
        return max(self.data.n_train // self.data.batch_size, 1)


# Alias for backward compatibility
NSFNO2DConfig = Default