import torch
import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.layers.channel_mlp import ChannelMLP as SciChannelMLP
from neuralop.layers.channel_mlp import ChannelMLP as TorchChannelMLP


def test_channel_mlp_parity():
    torch.manual_seed(0)

    # Input (PyTorch format)
    
    x_torch = torch.randn(1, 16, 64) 
    
    

    # Convert to JAX format
    
    x_jax = jnp.array(x_torch.numpy())
    print("\n--- INPUT CHECK ---")
    print("Torch input (first 5x5):\n", x_torch[0, :5, :5])
    print("JAX input (first 5x5):\n", x_jax[0, :5, :5])
    x_jax = jnp.transpose(x_jax, (0, 2, 1))  # (B, S, C)
    

    # Models
    
    torch_model = TorchChannelMLP(
        in_channels=16,
        out_channels=16,
        hidden_channels=16,
        n_layers=2,
        dropout=0.0
    )

    jax_model = SciChannelMLP(
        out_channels=16,
        hidden_channels=16,
        n_layers=2,
        dropout_rate=0.0
    )


    # Init JAX params
    
    key = jax.random.PRNGKey(0)
    params = jax_model.init(key, x_jax)

   
    # Copy weights
 
    for i, torch_layer in enumerate(torch_model.fcs):
        torch_layer = torch_model.fcs[i]
        jax_layer = f"dense_{i}"

        # Weight
        w = torch_layer.weight.detach().numpy()  # (out, in, 1)
        w = np.squeeze(w, axis=-1)               # (out, in)
        w = w.T                                  # (in, out)

        # Bias
        b = torch_layer.bias.detach().numpy()

        # Assign to JAX params
        params["params"][jax_layer]["kernel"] = jnp.array(w)
        params["params"][jax_layer]["bias"] = jnp.array(b)

    # Forward pass

    out_torch = torch_model(x_torch)

    out_jax = jax_model.apply(params, x_jax)

    # Convert JAX output → PyTorch format
    out_jax = np.array(out_jax)
    out_jax = np.transpose(out_jax, (0, 2, 1))

    out_torch = out_torch.detach().numpy()

    # Compare

    print("Torch shape:", out_torch.shape)
    print("JAX shape:", out_jax.shape)

    diff = np.mean(np.abs(out_torch - out_jax))
    print("Mean diff:", diff)

    max_diff = np.max(np.abs(out_torch - out_jax))
    print("Max diff:", max_diff)
    
    print("\n--- OUTPUT CHECK ---")
    print("Torch output (first 5x5):\n", out_torch[0, :5, :5])
    print("JAX output (first 5x5):\n", out_jax[0, :5, :5])
    
    assert np.allclose(out_torch, out_jax, atol=1e-3)

    print("ChannelMLP parity PASSED")
