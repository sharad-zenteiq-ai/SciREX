import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.segment_csr import segment_sum, segment_mean


@pytest.mark.parametrize(
    "N, K, C",
    [
        (5, 3, 2),
        (10, 4, 8),
    ],
)
def test_segment_sum_shape(N, K, C):
    edge_values = jnp.ones((N * K, C))
    segment_ids = jnp.repeat(jnp.arange(N), K)

    out = segment_sum(edge_values, segment_ids, num_segments=N)
    assert out.shape == (N, C)
    assert jnp.allclose(out, K)


def test_segment_mean_masked():
    # 2 target nodes, 3 neighbors each
    N, K, C = 2, 3, 1
    
    # Values:
    # node 0 gets: 10.0, 20.0, 30.0 (all valid)
    # node 1 gets: 10.0, 20.0, 30.0 (only first 2 valid)
    edge_values = jnp.array([
        [10.0], [20.0], [30.0],
        [10.0], [20.0], [30.0]
    ])
    
    segment_ids = jnp.repeat(jnp.arange(N), K)
    
    mask = jnp.array([
        True, True, True,
        True, True, False
    ])
    
    # We apply the mask *before* segment_mean in the layer logic (rep * mask_val)
    # Let's simulate that
    masked_edge_values = edge_values * mask[:, None]
    
    out = segment_mean(masked_edge_values, segment_ids, num_segments=N, mask=mask)
    
    assert out.shape == (2, 1)
    
    # Node 0 mean: (10+20+30)/3 = 20
    assert jnp.isclose(out[0, 0], 20.0)
    
    # Node 1 mean: (10+20)/2 = 15.0  (because mask=False for 30.0)
    assert jnp.isclose(out[1, 0], 15.0)


def test_segment_ops_jit():
    N, K, C = 4, 3, 2
    edge_values = jnp.ones((N * K, C))
    segment_ids = jnp.repeat(jnp.arange(N), K)
    mask = jnp.ones((N * K,))

    @jax.jit
    def run_sum(val, ids):
        return segment_sum(val, ids, num_segments=N)

    @jax.jit
    def run_mean(val, ids, m):
        return segment_mean(val, ids, num_segments=N, mask=m)

    out_sum = run_sum(edge_values, segment_ids)
    out_mean = run_mean(edge_values, segment_ids, mask)

    assert out_sum.shape == (N, C)
    assert out_mean.shape == (N, C)

def test_segment_mean_zero_division():
    N, K, C = 1, 3, 1
    edge_values = jnp.zeros((N * K, C))
    segment_ids = jnp.repeat(jnp.arange(N), K)
    mask = jnp.zeros((N * K,), dtype=bool)

    out = segment_mean(edge_values, segment_ids, num_segments=N, mask=mask)
    
    # Should not produce NaNs
    assert not jnp.isnan(out).any()
    assert jnp.allclose(out, 0.0)
