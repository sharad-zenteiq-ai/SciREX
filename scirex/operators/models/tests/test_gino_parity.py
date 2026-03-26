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

import os
import sys
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
import numpy as np
import pytest
from flax import linen as fnn

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from scirex.operators.models.gino import GINO as JaxGINO
from neuralop.models import GINO as PtGINO
from scirex.operators.layers.neighbor_search import NeighborSearch as JAX_NeighborSearch

# ==============================================================================
# WEIGHT MAPPING UTILITIES (REUSED FROM FNOGNO)
# ==============================================================================

def map_channel_mlp(jax_params, pt_module):
    """Maps JAX ChannelMLP parameters to PyTorch ChannelMLP module."""
    new_state_dict = {}
    for i in range(len(pt_module.fcs)):
        jax_dense = jax_params[f'dense_{i}']
        # neuralop.layers.channel_mlp.ChannelMLP uses Conv1d with kernel_size=1
        w = torch.from_numpy(np.array(jax_dense['kernel'].T)).to(torch.float64)
        if w.ndim == 2:
            w = w.unsqueeze(-1)
        new_state_dict[f'fcs.{i}.weight'] = w
        new_state_dict[f'fcs.{i}.bias'] = torch.from_numpy(np.array(jax_dense['bias'])).to(torch.float64)
    pt_module.load_state_dict(new_state_dict, strict=True)

def map_fno_blocks(jax_params, pt_module):
    """Maps JAX FNOBlocks parameters to PyTorch FNOBlocks module."""
    new_state_dict = {}
    for i in range(pt_module.n_layers):
        jax_block = jax_params[f'FNOBlock_{i}']
        
        # 1. Spectral Conv
        jax_conv = jax_block['conv']
        n_modes = pt_module.convs[i].n_modes
        for m_idx in range(len(n_modes)):
            w_real = np.array(jax_conv[f'weight_{m_idx}']['real'])
            w_imag = np.array(jax_conv[f'weight_{m_idx}']['imag'])
            w_torch = torch.complex(torch.from_numpy(w_real), torch.from_numpy(w_imag)).to(torch.float64)
            new_state_dict[f'convs.{i}.weight_{m_idx}'] = w_torch
            
        # 2. FNO Skip
        jax_skip = jax_block['skip']
        new_state_dict[f'fno_skips.{i}.weight'] = torch.from_numpy(np.array(jax_skip['kernel'].T)).to(torch.float64)
        if 'bias' in jax_skip:
            new_state_dict[f'fno_skips.{i}.bias'] = torch.from_numpy(np.array(jax_skip['bias'])).to(torch.float64)
            
        # 3. Channel MLP
        if hasattr(pt_module, 'mlp') and pt_module.mlp:
            jax_mlp = jax_block['channel_mlp']
            # mlp.0.fcs.0.weight
            for l_idx in range(len(pt_module.mlp[i].fcs)):
                jax_dense = jax_mlp[f'fcs_{l_idx}']
                new_state_dict[f'mlp.{i}.fcs.{l_idx}.weight'] = torch.from_numpy(np.array(jax_dense['kernel'].T)).to(torch.float64)
                new_state_dict[f'mlp.{i}.fcs.{l_idx}.bias'] = torch.from_numpy(np.array(jax_dense['bias'])).to(torch.float64)
                
            jax_mlp_skip = jax_block['channel_mlp_skip']
            new_state_dict[f'mlp_skips.{i}.weight'] = torch.from_numpy(np.array(jax_mlp_skip['kernel'].T)).to(torch.float64)
            if 'bias' in jax_mlp_skip:
                new_state_dict[f'mlp_skips.{i}.bias'] = torch.from_numpy(np.array(jax_mlp_skip['bias'])).to(torch.float64)

    pt_module.load_state_dict(new_state_dict, strict=False)

def map_gno_block(jax_params, pt_module):
    """Maps JAX GNOBlock parameters to PyTorch GNOBlock module."""
    new_state_dict = {}
    jax_it_params = jax_params['integral_transform']
    jax_mlp_params = jax_it_params['channel_mlp']
    
    for l_idx in range(len(pt_module.integral_transform.channel_mlp.fcs)):
        jax_dense = jax_mlp_params[f'dense_{l_idx}']
        # LinearChannelMLP uses nn.Linear
        new_state_dict[f'integral_transform.channel_mlp.fcs.{l_idx}.weight'] = torch.from_numpy(np.array(jax_dense['kernel'].T)).to(torch.float64)
        new_state_dict[f'integral_transform.channel_mlp.fcs.{l_idx}.bias'] = torch.from_numpy(np.array(jax_dense['bias'])).to(torch.float64)
        
    pt_module.load_state_dict(new_state_dict, strict=False)

def map_jax_to_pt_gino(jax_params, pt_model):
    """Full mapping for GINO model mapping to JAX default naming."""
    # GNOBlock_0 is gno_in
    map_gno_block(jax_params['GNOBlock_0'], pt_model.gno_in)
    
    # ChannelMLP_0 is lifting
    map_channel_mlp(jax_params['ChannelMLP_0'], pt_model.lifting)

    # FNO blocks are direct children FNOBlock_0, FNOBlock_1
    new_state_dict = {}
    for i in range(pt_model.fno_blocks.n_layers):
        jax_block = jax_params[f'FNOBlock_{i}']
        
        # 1. Spectral Conv
        jax_conv = jax_block['SpectralConv_0']
        n_modes = pt_model.fno_blocks.convs[i].n_modes
        for m_idx in range(len(n_modes)):
            w_real = np.array(jax_conv[f'weight_{m_idx}']['real'])
            w_imag = np.array(jax_conv[f'weight_{m_idx}']['imag'])
            w_torch = torch.complex(torch.from_numpy(w_real), torch.from_numpy(w_imag)).to(torch.float64)
            new_state_dict[f'fno_blocks.convs.{i}.weight_{m_idx}'] = w_torch
            
        # 2. FNO Skip
        jax_skip = jax_block['SkipConnection_0']
        new_state_dict[f'fno_blocks.fno_skips.{i}.weight'] = torch.from_numpy(np.array(jax_skip['kernel'].T)).to(torch.float64)
        if 'bias' in jax_skip:
            new_state_dict[f'fno_blocks.fno_skips.{i}.bias'] = torch.from_numpy(np.array(jax_skip['bias'])).to(torch.float64)
            
        # 3. Channel MLP
        if hasattr(pt_model.fno_blocks, 'channel_mlp') and pt_model.fno_blocks.channel_mlp:
            jax_mlp = jax_block['ChannelMLP_0']
            # pt_model.fno_blocks.channel_mlp is a ModuleList of ChannelMLP
            pt_mlp_module = pt_model.fno_blocks.channel_mlp[i]
            for l_idx in range(len(pt_mlp_module.fcs)):
                jax_dense = jax_mlp[f'dense_{l_idx}']
                # ChannelMLP inside FNOBlock in neuralop uses Conv1d too
                w = torch.from_numpy(np.array(jax_dense['kernel'].T)).to(torch.float64)
                if w.ndim == 2:
                    w = w.unsqueeze(-1)
                new_state_dict[f'fno_blocks.channel_mlp.{i}.fcs.{l_idx}.weight'] = w
                new_state_dict[f'fno_blocks.channel_mlp.{i}.fcs.{l_idx}.bias'] = torch.from_numpy(np.array(jax_dense['bias'])).to(torch.float64)
                
            jax_mlp_skip = jax_block['SkipConnection_1']
            new_state_dict[f'fno_blocks.channel_mlp_skips.{i}.weight'] = torch.from_numpy(np.array(jax_mlp_skip['kernel'].T)).to(torch.float64)
            if 'bias' in jax_mlp_skip:
                new_state_dict[f'fno_blocks.channel_mlp_skips.{i}.bias'] = torch.from_numpy(np.array(jax_mlp_skip['bias'])).to(torch.float64)

    pt_model.load_state_dict(new_state_dict, strict=False)
    
    # GNOBlock_1 is gno_out
    map_gno_block(jax_params['GNOBlock_1'], pt_model.gno_out)
    
    # ChannelMLP_1 is projection
    map_channel_mlp(jax_params['ChannelMLP_1'], pt_model.projection)

# ==============================================================================
# TEST CASE
# ==============================================================================

def test_gino_parity():
    # Use float64 for parity
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    
    seed = 42
    batch_size = 1
    nx, ny, nz = 8, 8, 8
    n_in = 32
    n_out = 16
    in_channels = 3
    out_channels = 1
    coord_dim = 3
    
    # ── 1. Create Inputs ──
    rng = jax.random.PRNGKey(seed)
    rng_f, rng_in_geom, rng_out_queries, rng_init = jax.random.split(rng, 4)
    
    # Grid
    x_grid = jnp.linspace(0, 1, nx, dtype=jnp.float64)
    y_grid = jnp.linspace(0, 1, ny, dtype=jnp.float64)
    z_grid = jnp.linspace(0, 1, nz, dtype=jnp.float64)
    grid = jnp.stack(jnp.meshgrid(x_grid, y_grid, z_grid, indexing='ij'), axis=-1)
    latent_queries_jax = jnp.repeat(grid[None, ...], batch_size, axis=0) # (b, 8, 8, 8, 3)
    
    input_geom_jax = jax.random.normal(rng_in_geom, (n_in, coord_dim), dtype=jnp.float64)
    x_in_jax = jax.random.normal(rng_f, (batch_size, n_in, in_channels), dtype=jnp.float64)
    output_queries_jax = jax.random.uniform(rng_out_queries, (n_out, coord_dim), dtype=jnp.float64)
    
    # ── 2. JAX Model ──
    jax_model = JaxGINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=coord_dim,
        in_gno_radius=0.5,
        out_gno_radius=0.5,
        fno_n_modes=(4, 4, 4),
        fno_hidden_channels=16,
        fno_n_layers=2,
        in_gno_channel_mlp_hidden_layers=(16, 16),
        out_gno_channel_mlp_hidden_layers=(16, 16),
        max_neighbors=20
    )
    
    # Initialized weights
    variables = jax_model.init(
        rng_init, 
        input_geom=input_geom_jax, 
        latent_queries=latent_queries_jax,
        output_queries=output_queries_jax,
        x=x_in_jax
    )
    print("\nJAX GINO Params Keys:", variables['params'].keys())
    
    # Neighbors search
    ns = JAX_NeighborSearch(max_neighbors=20)
    # IN GNO: Points to Grid
    in_nb = ns(points=input_geom_jax, queries=grid.reshape(-1, 3), radius=0.5)
    # OUT GNO: Grid to Points
    out_nb = ns(points=grid.reshape(-1, 3), queries=output_queries_jax, radius=0.5)
    
    jax_out = jax_model.apply(
        variables, 
        input_geom=input_geom_jax, 
        latent_queries=latent_queries_jax,
        output_queries=output_queries_jax,
        x=x_in_jax,
        in_neighbors=in_nb,
        out_neighbors=out_nb
    )
    
    # ── 3. PyTorch Model ──
    pt_model = PtGINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=coord_dim,
        in_gno_radius=0.5,
        out_gno_radius=0.5,
        fno_n_modes=(4, 4, 4),
        fno_hidden_channels=16,
        fno_n_layers=2,
        in_gno_channel_mlp_hidden_layers=[16, 16],
        out_gno_channel_mlp_hidden_layers=[16, 16],
        gno_use_torch_scatter=False,
        gno_use_open3d=False
    ).to(torch.float64)
    
    # Map weights
    map_jax_to_pt_gino(variables['params'], pt_model)
    pt_model.eval()
    
    # Prepare PT inputs
    in_geom_pt = torch.from_numpy(np.array(input_geom_jax)).to(torch.float64)
    queries_grid_pt = torch.from_numpy(np.array(grid[None, ...])).to(torch.float64)
    output_queries_pt = torch.from_numpy(np.array(output_queries_jax)).to(torch.float64)
    x_pt = torch.from_numpy(np.array(x_in_jax)).to(torch.float64)
    
    with torch.no_grad():
        pt_out = pt_model(
            input_geom=in_geom_pt,
            latent_queries=queries_grid_pt,
            output_queries=output_queries_pt,
            x=x_pt
        )
        
    # ── 4. Compare ──
    jax_out_np = np.array(jax_out)
    pt_out_np = pt_out.numpy()
    
    np.testing.assert_allclose(jax_out_np, pt_out_np, rtol=1e-3, atol=1e-3)
    print("GINO Model Parity PASSED!")

if __name__ == "__main__":
    test_gino_parity()
