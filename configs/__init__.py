# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform).
#
# Licensed under the Apache License, Version 2.0

"""
SciREX configuration presets.

Model architecture presets  (configs.models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :class:`FNOConfig`        – full FNO parameter surface
- :class:`SimpleFNOConfig`  – simplified subset
- :class:`FNO_Small2D`      – small 2D FNO
- :class:`FNO_Medium2D`     – medium 2D FNO (default)
- :class:`FNO_Large2D`      – large 2D FNO
- :class:`FNO_Small3D`      – small 3D FNO
- :class:`FNO_Medium3D`     – medium 3D FNO (default)

Experiment configs  (model + training + data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :class:`FNO2DConfig`      – 2D Poisson experiment
- :class:`FNO3DConfig`      – 3D Poisson experiment
"""

# ── Model architecture presets ──
from configs.models import (  # noqa: F401
    ModelConfig,
    FNOConfig,
    SimpleFNOConfig,
    FNO_Small2D,
    FNO_Medium2D,
    FNO_Large2D,
    FNO_Small3D,
    FNO_Medium3D,
)

# ── Experiment configs ──
from configs.poisson_fno_config import FNO2DConfig, FNO3DConfig  # noqa: F401
from configs.ns_fno3d_config import (
    NSFNO3DConfig,
    NSFNO3D_Small,
    NSFNO3D_Medium,
    NSFNO3D_Large,
    NSFNO3D_Huge,
)  # noqa: F401
