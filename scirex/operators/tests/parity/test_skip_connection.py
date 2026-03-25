import torch
import jax
import jax.numpy as jnp
import numpy as np

from scirex.operators.layers.skip_connection import SkipConnection as SciSkip
from neuralop.layers.skip_connections import skip_connection as TorchSkip


def test_skip_identity():
    torch.manual_seed(0)

    x_torch = torch.randn(1, 16, 64)

    # JAX input
    x_jax = jnp.array(x_torch.numpy())
    x_jax = jnp.transpose(x_jax, (0, 2, 1))

    # Models
    torch_model = TorchSkip(16, 16, skip_type="identity")
    jax_model = SciSkip(out_channels=16, skip_type="identity")

    # Forward
    out_torch = torch_model(x_torch)
    out_jax = jax_model.apply({}, x_jax)

    out_jax = np.transpose(np.array(out_jax), (0, 2, 1))
    out_torch = out_torch.detach().numpy()

    assert np.allclose(out_torch, out_jax)


def test_skip_linear():
    torch.manual_seed(0)

    x_torch = torch.randn(1, 16, 64)

    x_jax = jnp.array(x_torch.numpy())
    x_jax = jnp.transpose(x_jax, (0, 2, 1))

    torch_model = TorchSkip(16, 16, skip_type="linear")
    jax_model = SciSkip(out_channels=16, skip_type="linear")

    # Init JAX
    key = jax.random.PRNGKey(0)
    params = jax_model.init(key, x_jax)

    # Copy weights
    w = torch_model.conv.weight.detach().numpy()  # (out, in, 1)
    w = np.squeeze(w, axis=-1)
    w = w.T

    params["params"]["Dense_0"]["kernel"] = jnp.array(w)

    # Forward
    out_torch = torch_model(x_torch)
    out_jax = jax_model.apply(params, x_jax)

    out_jax = np.transpose(np.array(out_jax), (0, 2, 1))
    out_torch = out_torch.detach().numpy()

    assert np.allclose(out_torch, out_jax, atol=1e-3)


def test_skip_soft_gating():
    torch.manual_seed(0)

    x_torch = torch.randn(1, 16, 64)

    x_jax = jnp.array(x_torch.numpy())
    x_jax = jnp.transpose(x_jax, (0, 2, 1))

    torch_model = TorchSkip(16, 16, skip_type="soft-gating")
    jax_model = SciSkip(out_channels=16, skip_type="soft-gating")

    key = jax.random.PRNGKey(0)
    params = jax_model.init(key, x_jax)

    # Copy weight
    w = torch_model.weight.detach().numpy()  # (1, C, 1, 1)
    w = np.squeeze(w)       # (C,)
    w = w.reshape(1, 1, -1) # (1, 1, C)

    params["params"]["SoftGating_0"]["weight"] = jnp.array(w)

    # Forward
    out_torch = torch_model(x_torch)
    out_jax = jax_model.apply(params, x_jax)

    out_jax = np.transpose(np.array(out_jax), (0, 2, 1))
    out_torch = out_torch.detach().numpy()

    assert np.allclose(out_torch, out_jax, atol=1e-3)