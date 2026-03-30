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
"""

from typing import Optional, Literal, Tuple, Any
from zencfg import ConfigBase

# Base config
class ModelConfig(ConfigBase):
    """Base configuration for any neural operator model."""
    arch: str = "fno"
    in_channels: int = 1
    out_channels: int = 1


# Full FNO config 
class FNOConfig(ModelConfig):
    """
    Complete FNO architecture configuration.
    """
    arch: str = "fno"
    n_modes: Tuple[int, ...] = (16, 16)
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


# Simplified FNO config
class SimpleFNOConfig(FNOConfig):
    """
    Exposes only the most commonly tuned FNO parameters.
    """
    pass


class FNO_Small2D(SimpleFNOConfig):
    n_modes: Tuple[int, ...] = (16, 16)
    hidden_channels: int = 24
    n_layers: int = 4
    in_channels: int = 3
    out_channels: int = 1
    use_grid: bool = False


class FNO_Medium2D(SimpleFNOConfig):
    n_modes: Tuple[int, ...] = (24, 24)
    hidden_channels: int = 128
    n_layers: int = 4
    in_channels: int = 3
    out_channels: int = 1
    use_grid: bool = False
    use_norm: bool = True


class FNO_Large2D(SimpleFNOConfig):
    n_modes: Tuple[int, ...] = (32, 32)
    hidden_channels: int = 256
    n_layers: int = 4
    in_channels: int = 3
    out_channels: int = 1
    use_grid: bool = False
    use_norm: bool = True


class FNO_Small3D(SimpleFNOConfig):
    n_modes: Tuple[int, ...] = (8, 8, 8)
    hidden_channels: int = 24
    n_layers: int = 4
    in_channels: int = 4
    out_channels: int = 1
    use_grid: bool = False
    use_norm: bool = True


class FNO_Medium3D(SimpleFNOConfig):
    n_modes: Tuple[int, ...] = (16, 16, 16)
    hidden_channels: int = 128
    n_layers: int = 4
    in_channels: int = 4
    out_channels: int = 1
    use_grid: bool = False
    use_norm: bool = True


#  G I N O   
class GINOConfig(ModelConfig):
    """
    Complete GINO architecture configuration.
    """
    arch: str = "gino"

    # GINO core
    latent_feature_channels: Optional[int] = None
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 3
    in_gno_radius: float = 0.033
    out_gno_radius: float = 0.033
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"

    in_gno_pos_embed_type: str = "transformer"
    out_gno_pos_embed_type: str = "transformer"
    gno_embed_channels: int = 32
    gno_embed_max_positions: int = 10000

    # FNO parameters
    fno_n_modes: Tuple[int, ...] = (16, 16, 16)
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 2
    fno_n_layers: int = 4

    # GNO MLP parameters
    in_gno_channel_mlp_hidden_layers: Tuple[int, ...] = (80, 80, 80)
    out_gno_channel_mlp_hidden_layers: Tuple[int, ...] = (512, 256)

    # FNO extras
    fno_use_channel_mlp: bool = True
    fno_norm: bool = False
    fno_skip: Literal["identity", "linear", "soft-gating"] = "linear"
    fno_channel_mlp_skip: Literal["identity", "linear", "soft-gating"] = "soft-gating"

    max_neighbors: int = 64
    use_neighbor_cache: bool = True


class GINO_Small3d(GINOConfig):
    in_channels: int = 1
    out_channels: int = 1
    fno_n_modes: Tuple[int, ...] = (8, 8, 8)
    fno_hidden_channels: int = 64
    fno_n_layers: int = 4
    in_gno_radius: float = 0.05
    out_gno_radius: float = 0.05


#  F N O G N O   
class FNOGNOConfig(ModelConfig):
    """
    Complete FNOGNO architecture configuration.
    """
    arch: str = "fnogno"

    # FNOGNO core
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 3
    gno_radius: float = 0.033
    gno_transform_type: str = "linear"

    gno_pos_embed_type: str = "transformer"
    gno_embed_channels: int = 32
    gno_embed_max_positions: int = 10000

    # FNO parameters
    fno_n_modes: Tuple[int, ...] = (16, 16, 16)
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 4
    fno_n_layers: int = 4

    # GNO MLP parameters
    gno_channel_mlp_hidden_layers: Tuple[int, ...] = (512, 256)

    # FNO extras
    fno_use_channel_mlp: bool = True
    fno_norm: bool = False
    fno_skip: Literal["identity", "linear", "soft-gating"] = "linear"
    fno_channel_mlp_skip: Literal["identity", "linear", "soft-gating"] = "soft-gating"

    max_neighbors: int = 64
    use_neighbor_cache: bool = True


class FNOGNO_Small3d(FNOGNOConfig):
    in_channels: int = 1
    out_channels: int = 1
    fno_n_modes: Tuple[int, ...] = (8, 8, 8)
    fno_hidden_channels: int = 64
    fno_n_layers: int = 4
    gno_radius: float = 0.05