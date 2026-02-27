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
Model architecture configuration presets for SciREX neural operators.

Pure architecture configs — no training or data parameters.
Inspired by neuraloperator (https://github.com/neuraloperator/neuraloperator).

Hierarchy
---------
ModelConfig                   -- abstract base for any neural operator
  └── FNOConfig               -- full FNO parameter surface
        └── SimpleFNOConfig   -- simplified subset (most common knobs)
              ├── FNO_Small2D
              ├── FNO_Medium2D
              ├── FNO_Large2D
              ├── FNO_Small3D
              └── FNO_Medium3D

Usage
-----
    from configs.models import FNO_Small2D

    cfg = FNO_Small2D()
    model = FNO2D(
        hidden_channels=cfg.hidden_channels,
        n_layers=cfg.n_layers,
        n_modes=cfg.n_modes,
        out_channels=cfg.out_channels,
    )
"""

from dataclasses import dataclass
from typing import Optional, Literal


# ────────────────────────────────────────────────────────────────────
# Base config
# ────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Base configuration for any neural operator model."""
    arch: str = "fno"
    in_channels: int = 1
    out_channels: int = 1


# ────────────────────────────────────────────────────────────────────
# Full FNO config  (exposes every tuneable architecture knob)
# ────────────────────────────────────────────────────────────────────
@dataclass
class FNOConfig(ModelConfig):
    """
    Complete FNO architecture configuration.

    Parameters
    ----------
    n_modes : tuple of int
        Fourier modes to keep per spatial dimension.
        Length determines 2D vs 3D.
    hidden_channels : int
        Width of the spectral layers.
    n_layers : int
        Number of spectral blocks.
    in_channels : int
        Input channels (e.g. 1 for a scalar field).
    out_channels : int
        Output channels.
    lifting_channel_ratio : int
        Ratio of lifting channels to hidden_channels.
    projection_channel_ratio : int
        Ratio of projection channels to hidden_channels.
    use_grid : bool
        Whether to automatically append normalized spatial coordinates.
    fno_skip : str
        Type of skip connection for the spectral branch.
    channel_mlp_skip : str
        Type of skip connection for the channel MLP branch.
    use_channel_mlp : bool
        Whether to use a channel MLP refinement in each block.
    domain_padding : float
        Fraction of zero-padding applied via ``DomainPadding``.
        Used by both ``FNO2D`` and ``FNO3D``.
    use_norm : bool
        Whether to apply instance normalization inside each
        ``SpectralBlock``.
    """
    arch: str = "fno"
    n_modes: tuple = (16, 16)
    hidden_channels: int = 64
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 4
    use_grid: bool = True
    fno_skip: Literal["identity", "linear", "soft-gating"] = "linear"
    channel_mlp_skip: Literal["identity", "linear", "soft-gating"] = "soft-gating"
    use_channel_mlp: bool = True
    domain_padding: float = 0.0
    use_norm: bool = False


# ────────────────────────────────────────────────────────────────────
# Simplified FNO config  (the knobs you tweak 90 % of the time)
# ────────────────────────────────────────────────────────────────────
@dataclass
class SimpleFNOConfig(FNOConfig):
    """
    SimpleFNOConfig: exposes only the most commonly tuned FNO
    parameters while inheriting sensible defaults for the rest.

    This is the recommended base class for defining quick presets.
    """
    n_modes: tuple = (16, 16)
    hidden_channels: int = 64
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1


# ═══════════════════════════════════════════════════════════════════
#  2-D   P R E S E T S
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FNO_Small2D(SimpleFNOConfig):
    """
    A small FNO for 2D problems.

    Good for quick prototyping and sanity checks.  ~30 k params.
    """
    n_modes: tuple = (16, 16)
    hidden_channels: int = 24
    n_layers: int = 4
    in_channels: int = 3
    out_channels: int = 1
    use_grid: bool = False


@dataclass
class FNO_Medium2D(SimpleFNOConfig):
    """
    A medium-sized FNO for 2D problems.

    The default workhorse for most 2D operator-learning tasks such
    as Poisson, Darcy, etc.
    """
    n_modes: tuple = (24, 24)
    hidden_channels: int = 128
    n_layers: int = 4
    in_channels: int = 3
    out_channels: int = 1
    use_grid: bool = False
    use_norm: bool = True


@dataclass
class FNO_Large2D(SimpleFNOConfig):
    """
    A large FNO for 2D problems.

    More Fourier modes and wider hidden channels for higher-resolution
    or harder 2D problems.
    """
    n_modes: tuple = (32, 32)
    hidden_channels: int = 256
    n_layers: int = 4
    in_channels: int = 3
    out_channels: int = 1
    use_grid: bool = False
    use_norm: bool = True


# ═══════════════════════════════════════════════════════════════════
#  3-D   P R E S E T S
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FNO_Small3D(SimpleFNOConfig):
    """
    A small FNO for 3D problems.

    Suitable for low-resolution 3D experiments on a single GPU.
    """
    n_modes: tuple = (8, 8, 8)
    hidden_channels: int = 24
    n_layers: int = 4
    in_channels: int = 4
    out_channels: int = 1
    use_grid: bool = False
    use_norm: bool = True


@dataclass
class FNO_Medium3D(SimpleFNOConfig):
    """
    A medium FNO for 3D problems.

    Default for 3D operator tasks such as 3D Poisson.
    Includes instance norm and a positional embedding for stability.
    """
    n_modes: tuple = (12, 12, 12)
    hidden_channels: int = 64
    n_layers: int = 4
    in_channels: int = 4
    out_channels: int = 1
    use_grid: bool = False
    use_norm: bool = True
