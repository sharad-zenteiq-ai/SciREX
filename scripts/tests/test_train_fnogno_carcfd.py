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

import pytest
import jax.numpy as jnp
import numpy as np
import optax
import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Workaround for testing functions from an un-importable script:
# Since the script contains a main() block, we import the functions from it manually.
# Another option is to refactor these functions into a separate utils file.
import importlib.util
spec = importlib.util.spec_from_file_location("train_script", os.path.join(project_root, "scripts/train_fnogno_carcfd.py"))
train_fnogno = importlib.util.module_from_spec(spec)
# Note: This might trigger the main() block if it's not guarded by if __name__ == "__main__".
# Let's check the script again.
spec.loader.exec_module(train_fnogno)

from configs.fnogno_carcfd_config import FNOGNOCarCFDConfig

def test_make_schedule():
    """Test learning rate schedule creation."""
    config = FNOGNOCarCFDConfig()
    steps_per_epoch = 10
    schedule = train_fnogno.make_schedule(config, steps_per_epoch)
    
    # Check that it's a callable
    assert callable(schedule)
    
    # Check initial value
    lr_init = schedule(0)
    assert lr_init == 0.0 # Warmup starts at 0
    
    # Check value after warmup
    total_steps = config.opt.n_epochs * steps_per_epoch
    warmup_steps = min(310, total_steps // 10)
    lr_at_peak = schedule(warmup_steps)
    assert np.allclose(lr_at_peak, config.opt.learning_rate)

def test_preprocess_cfd_sample():
    """Test preprocessing of CFD data batches."""
    config = FNOGNOCarCFDConfig()
    
    dummy_batch = {
        "vertices": np.random.randn(1, 100, 3),
        "query_points": np.random.randn(1, 8, 8, 8, 3),
        "distance": np.random.randn(1, 8, 8, 8, 1),
        "press": np.random.randn(1, 100, 1),
        "in_neighbors": {"indices": np.zeros((1, 512, 10))},
        "out_neighbors": {"indices": np.zeros((1, 100, 12))}
    }
    
    processed = train_fnogno.preprocess_cfd_sample(dummy_batch, config)
    
    assert len(processed) == 7
    batch_geom, batch_queries, batch_out_queries, batch_lat_features, batch_y, batch_in_nb, batch_out_nb = processed
    
    assert batch_geom.shape == (1, 100, 3)
    assert batch_queries.shape == (1, 8, 8, 8, 3)
    assert batch_lat_features.shape == (1, 8, 8, 8, 1)
    assert batch_y.shape == (1, 100, 1)

if __name__ == "__main__":
    pytest.main([__file__])
