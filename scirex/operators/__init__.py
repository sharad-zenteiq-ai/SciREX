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
`scirex.operators` provides the core neural operator architectures and their 
constituent layers, including Fourier Neural Operators (FNO) and Wavelet 
Neural Operators (WNO).
"""

# ── Layers ──
from .layers.channel_mlp import ChannelMLP
from .layers.spectral_conv import SpectralConv2D, SpectralConv3D
from .layers.padding import DomainPadding
from .layers.embeddings import GridEmbedding
from .layers.skip_connection import SkipConnection, SoftGating
from .layers.fno_block import FNOBlock, FNOBlock3D
from .layers.wavelet_conv import WaveletConv1D, WaveletConv2D
from .layers.wavelet_block import WaveletBlock1D, WaveletBlock2D
from .layers.fast_wavelet_conv import LiftingWaveletConv2D

# ── Models ──
from .models.fno import FNO2D, FNO3D
from .models.wno import WNO1D, WNO2D, FWNO2D

# ── GNO / Future Layers ──
from .layers.integral_transform import IntegralTransform

# ── Data, Losses, Training ──
from . import data
from . import losses
from . import training
from . import config
