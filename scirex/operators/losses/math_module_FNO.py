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

import jax
import jax.numpy as jnp
from flax import linen as nn


def exact_spectral_derivative(x: jnp.ndarray, deriv_order: int) -> jnp.ndarray:
    """
    Computes the exact spatial derivative in the Fourier domain without learnable parameters.
    x shape: (batch, nx, ny, channels)
    """
    nx, ny = x.shape[1], x.shape[2]
    Lx, Ly = 1.0, 1.0  # Domain lengths
    
    # 1. Forward FFT
    x_ft = jnp.fft.rfftn(x, axes=(1, 2), norm="ortho")
    
    # 2. X-derivative multiplier
    k_x = jnp.fft.fftfreq(nx) * nx
    k_broadcast_x = ((2.0 * jnp.pi / Lx) * k_x).reshape((1, nx, 1, 1))
    mult_x = (1j * k_broadcast_x) ** deriv_order
    
    # 3. Y-derivative multiplier
    k_y = jnp.fft.rfftfreq(ny) * ny
    k_broadcast_y = ((2.0 * jnp.pi / Ly) * k_y).reshape((1, 1, ny // 2 + 1, 1))
    mult_y = (1j * k_broadcast_y) ** deriv_order
    
    # 4. Apply multipliers
    out_ft_x = x_ft * mult_x
    out_ft_y = x_ft * mult_y
    
    # 5. Inverse FFT
    dv_dx = jnp.fft.irfftn(out_ft_x, s=(nx, ny), axes=(1, 2), norm="ortho")
    dv_dy = jnp.fft.irfftn(out_ft_y, s=(nx, ny), axes=(1, 2), norm="ortho")
    
    # Stack resulting shape: (batch_size, nx, ny, channels, 2)
    return jnp.stack([dv_dx, dv_dy], axis=-1)



def first_order_derivative(params, v_L):
    # 1. Extract projection weights directly
    projection_params = params['projection_layer']
    W1 = projection_params['dense_0']['kernel']
    b1 = projection_params['dense_0']['bias']
    W2 = projection_params['dense_1']['kernel']
    
    # 2. Forward pass to hidden layer
    z = jnp.dot(v_L, W1) + b1
    
    # 3. 1st derivative of GELU
    flat_z = z.flatten()
    d_gelu = jax.vmap(jax.grad(nn.gelu))
    g_prime = d_gelu(flat_z).reshape(z.shape)
    
    # 4. Spectral derivatives
    v_prime = exact_spectral_derivative(v_L, deriv_order=1)
    v_prime_x, v_prime_y = v_prime[..., 0], v_prime[..., 1]
    
    # 5. Analytical forward-mode contraction
    # u_x = v'_x * W1
    u_x = jnp.dot(v_prime_x, W1)  
    u_y = jnp.dot(v_prime_y, W1)
    
    # Q'(v_L) * v' = (u_x * g') * W2
    u_prime_x = jnp.dot(u_x * g_prime, W2) 
    u_prime_y = jnp.dot(u_y * g_prime, W2)
    
    return jnp.stack([u_prime_x, u_prime_y], axis=-1)


def second_order_derivative(params, v_L):
    # 1. Extract projection weights directly
    projection_params = params['projection_layer']
    W1 = projection_params['dense_0']['kernel']
    b1 = projection_params['dense_0']['bias']
    W2 = projection_params['dense_1']['kernel']
    
    # 2. Forward pass to hidden layer
    z = jnp.dot(v_L, W1) + b1
    
    # 3. 1st and 2nd derivatives of GELU
    flat_z = z.flatten()
    d_gelu = jax.vmap(jax.grad(nn.gelu))
    d2_gelu = jax.vmap(jax.grad(jax.grad(nn.gelu)))
    
    g_prime = d_gelu(flat_z).reshape(z.shape)
    g_double_prime = d2_gelu(flat_z).reshape(z.shape)
    
    # 4. Spectral derivatives
    v_prime = exact_spectral_derivative(v_L, deriv_order=1)
    v_prime2 = exact_spectral_derivative(v_L, deriv_order=2)
    
    v_prime_x, v_prime_y = v_prime[..., 0], v_prime[..., 1]
    v_prime2_x, v_prime2_y = v_prime2[..., 0], v_prime2[..., 1]
    
    # -------------------------------------------------------------------
    # Term 1: Q''(v_L) * (v')^2
    # Analytical math: (v'_x W1)^2 * g'' * W2
    # -------------------------------------------------------------------
    u_x = jnp.dot(v_prime_x, W1)
    u_y = jnp.dot(v_prime_y, W1)
    
    term_1_x = jnp.dot((u_x ** 2) * g_double_prime, W2)
    term_1_y = jnp.dot((u_y ** 2) * g_double_prime, W2)
    
    # -------------------------------------------------------------------
    # Term 2: Q'(v_L) * v''
    # Analytical math: (v''_x W1) * g' * W2
    # -------------------------------------------------------------------
    w_x = jnp.dot(v_prime2_x, W1)
    w_y = jnp.dot(v_prime2_y, W1)
    
    term_2_x = jnp.dot(w_x * g_prime, W2)
    term_2_y = jnp.dot(w_y * g_prime, W2)
    
    # -------------------------------------------------------------------
    # Combine and stack
    # -------------------------------------------------------------------
    d2u_dx2 = term_1_x + term_2_x
    d2u_dy2 = term_1_y + term_2_y
    
    return jnp.stack([d2u_dx2, d2u_dy2], axis=-1)