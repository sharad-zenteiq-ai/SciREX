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
SciREX configuration presets.
"""

# ── Model architecture presets ──
from configs.models import (
    ModelConfig,
    FNOConfig,
    SimpleFNOConfig,
    FNO_Small2D,
    FNO_Medium2D,
    FNO_Large2D,
    FNO_Small3D,
    FNO_Medium3D,
    GINOConfig,
    GINO_Small3d,
    FNOGNOConfig,
    FNOGNO_Small3d,
)

# ── Experiment configs ──
from configs.poisson_fno_config import FNO2DConfig, FNO3DConfig
from configs.gino_carcfd_config import GINOCarCFDConfig
from configs.fnogno_carcfd_config import FNOGNOCarCFDConfig
