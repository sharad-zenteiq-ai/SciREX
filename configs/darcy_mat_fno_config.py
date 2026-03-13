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
Experiment config for FNO2D on the ORIGINAL FNO paper Darcy dataset.

Reproduces exactly: Li et al. 2021, arXiv:2010.08895, Table 1, Darcy row.
Original script:    fourier_neural_operator/fourier_2d.py

Available dataset files (confirmed on disk)
-------------------------------------------
    piececonst_r421_N1024_smooth1.mat   1692 MB   421x421   <- paper used this
    piececonst_r421_N1024_smooth2.mat   1692 MB   421x421
    piececonst_r241_N1024_smooth1.mat    412 MB   241x241   <- smaller, faster
    piececonst_r241_N1024_smooth2.mat    416 MB   241x241

Subsampling grid sizes
----------------------
    r421 files, stride r=5  ->  85x85   (paper default)
    r421 files, stride r=3  ->  141x141 (higher res)
    r241 files, stride r=3  ->  81x81   (close to paper, 3x faster to load)
    r241 files, stride r=1  ->  241x241 (full res of smaller file)

Paper hyperparameters (verbatim from fourier_2d.py)
----------------------------------------------------
    ntrain=1000, ntest=100, batch_size=20, lr=0.001
    epochs=500, step_size=100, gamma=0.5
    modes=12, width=32  (4 Fourier layers)
    grid: r421 subsampled at stride=5 -> 85x85

Expected results
----------------
    85x85,   n_train=1000  ->  Rel-L2 ~ 0.9%   (paper Table 1)
    81x81,   n_train=1000  ->  Rel-L2 ~ 1.0%   (similar)
    141x141, n_train=1000  ->  Rel-L2 ~ 0.8%   (slightly better)

Usage
-----
    from configs.darcy_mat_fno_config import DarcyMatFNO2DConfig

    # Paper-exact (uses r421, stride=5 -> 85x85):
    config = DarcyMatFNO2DConfig()

    # Faster (uses r241, stride=3 -> 81x81, loads 4x faster):
    config = DarcyMatFNO2DConfig(
        train_path="scirex/operators/data/darcy_original/piececonst_r241_N1024_smooth1.mat",
        test_path ="scirex/operators/data/darcy_original/piececonst_r241_N1024_smooth2.mat",
        orig_resolution=241,
        subsample_rate=3,
    )

    # Higher resolution (r421, stride=3 -> 141x141):
    config = DarcyMatFNO2DConfig(subsample_rate=3, n_modes_x=24, n_modes_y=24)
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class DarcyMatFNO2DConfig:
    """
    Paper-exact config for FNO2D on Darcy flow .mat files.
    All values taken directly from fourier_2d.py in the original repo.
    """

    # ── Dataset paths ─────────────────────────────────────────────────────
    train_path: str = "scirex/operators/data/darcy_original/piececonst_r421_N1024_smooth1.mat"
    test_path:  str = "scirex/operators/data/darcy_original/piececonst_r421_N1024_smooth2.mat"

    # Original full resolution of the .mat file grid
    # r421 files: orig_resolution=421
    # r241 files: orig_resolution=241
    orig_resolution: int = 421

    # Spatial stride when subsampling the full grid.
    # r=5 on r421 -> 85x85  (paper default)
    # r=3 on r421 -> 141x141
    # r=3 on r241 -> 81x81  (recommended if using r241 files)
    # r=1         -> full resolution (very slow)
    subsample_rate: int = 5

    # ── Dataset sizes ─────────────────────────────────────────────────────
    n_train: int = 1000   # paper used 1000
    n_test:  int = 100    # paper used 100

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MODEL — paper exact: modes=12, width=32, 4 Fourier layers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # "modes = 12" in fourier_2d.py
    # Nyquist cap for 85x85 grid = 42, so 12 is well within range
    n_modes_x: int = 12
    n_modes_y: int = 12

    # "width = 32" in fourier_2d.py  (hidden_channels in SciREX)
    hidden_channels: int = 32

    # 4 Fourier layers in the original
    n_layers: int = 4

    # in_channels = 3  (coeff + grid_x + grid_y)
    # out_channels = 1 (pressure)
    in_channels:  int = 3
    out_channels: int = 1

    # Architecture settings not in original paper but needed by SciREX FNO2D
    lifting_channel_ratio:    int   = 2
    projection_channel_ratio: int   = 4
    use_norm:                 bool  = False  # original paper had NO instance norm
    use_channel_mlp:          bool  = False
    domain_padding:           float = 0.0
    fno_skip:                 str   = "linear"
    channel_mlp_skip:         str   = "soft-gating"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRAINING — paper exact
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    learning_rate: float = 1e-3     # "lr = 0.001"
    weight_decay:  float = 1e-4
    batch_size:    int   = 20       # "batch_size = 20"
    epochs:        int   = 500      # "epochs = 500"
    steps_per_epoch: int = 50       # overridden at runtime: 1000//20 = 50

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LR SCHEDULE — paper used StepLR: halve every 100 epochs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    scheduler_type:      Literal["step", "cosine"] = "step"
    scheduler_step_size: int   = 100   # "step_size = 100"
    scheduler_gamma:     float = 0.5   # "gamma = 0.5"
    cosine_decay_epochs: int   = 500   # unused when scheduler_type="step"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NORMALISATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    encode_input:  bool = True
    encode_output: bool = True

    seed: int = 42

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def resolution(self) -> int:
        """Actual grid size after subsampling: ((orig-1)//r)+1."""
        return int(((self.orig_resolution - 1) / self.subsample_rate) + 1)

    @property
    def n_modes(self) -> Tuple[int, int]:
        """Fourier modes capped at Nyquist (resolution//2)."""
        cap = self.resolution // 2
        return (min(self.n_modes_x, cap), min(self.n_modes_y, cap))
