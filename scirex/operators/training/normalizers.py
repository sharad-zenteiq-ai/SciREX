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

import jax.numpy as jnp
import numpy as np

class GaussianNormalizer:
    def __init__(self, x, eps=1e-7):
        """
        x: (n_samples, nx, ny, channels) or any shape where we want to normalize across samples.
        We compute mean and std across the first dimension (n_samples) AND spatial dimensions?
        Usually, for FNO/WNO, normalization is done per channel across all samples and spatial locations.
        """
        # Compute stats over all but the last dimension (channels)
        # or maybe just the first? 
        # Most implementations in this field normalize per-channel across all spatial grid points.
        
        # Determine axes to reduce over (everything except the last channel dimension)
        reduce_axes = tuple(range(len(x.shape) - 1))
        
        self.mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        self.std = jnp.std(x, axis=reduce_axes, keepdims=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean
