import jax
import jax.numpy as jnp
import pytest

from scirex.operators.layers.neighbor_search import NeighborSearch

@pytest.mark.parametrize(
    "num_data, num_queries, dim, K",
    [
        (10, 5, 3, 2),
        (20, 8, 4, 3),
    ],
)
def test_neighbor_search_shapes(num_data, num_queries, dim, K):
    """NeighborSearch should return correct (N, K) shapes."""
    data = jnp.ones((num_data, dim))
    queries = jnp.ones((num_queries, dim))

    model = NeighborSearch(max_neighbors=K)
    out = model(points=data, queries=queries, radius=10.0)

    assert out["neighbor_indices"].shape == (num_queries, K)
    assert out["mask"].shape == (num_queries, K)

def test_neighbor_search_radius_mask():
    """Mask should correctly filter neighbors based on radius."""
    data = jnp.array([
        [0.0, 0.0],
        [10.0, 10.0],
    ])
    queries = jnp.array([
        [0.0, 0.0],
    ])
    model = NeighborSearch(max_neighbors=2)
    out = model(points=data, queries=queries, radius=1.0)
    
    mask = out["mask"][0]
    indices = out["neighbor_indices"][0]

    # Only first point should be within radius
    assert mask[0] == True
    assert mask[1] == False
    assert indices[1] == -1  # Padded element

def test_neighbor_search_with_distance():
    """Should return distances when return_norm=True."""
    data = jnp.array([
        [0.0, 0.0],
        [3.0, 4.0],  # distance = 5
    ])
    queries = jnp.array([
        [0.0, 0.0],
    ])
    model = NeighborSearch(max_neighbors=2, return_norm=True)
    out = model(points=data, queries=queries, radius=10.0)
    
    dists = out["distances"][0]
    assert dists.shape == (2,)
    assert jnp.all(dists >= 0)
    assert jnp.isclose(dists[1], 5.0)

def test_neighbor_search_pad_if_less_than_k():
    """Should pad with -1 if total points < K."""
    data = jnp.array([
        [0.0, 0.0],
    ])
    queries = jnp.array([
        [0.0, 0.0],
    ])
    model = NeighborSearch(max_neighbors=3, return_norm=True)
    out = model(points=data, queries=queries, radius=10.0)
    
    indices = out["neighbor_indices"][0]
    mask = out["mask"][0]
    distances = out["distances"][0]
    
    assert indices.shape == (3,)
    assert mask.shape == (3,)
    
    assert indices[0] == 0
    assert mask[0] == True
    
    assert indices[1] == -1
    assert mask[1] == False
    assert distances[1] == 0.0
    
    assert indices[2] == -1
    assert mask[2] == False
    assert distances[2] == 0.0

def test_neighbor_search_jit():
    """NeighborSearch should be JIT compatible."""
    data = jnp.ones((10, 3))
    queries = jnp.ones((5, 3))
    model = NeighborSearch(max_neighbors=2)

    @jax.jit
    def run(data, queries):
        return model(points=data, queries=queries, radius=5.0)

    out = run(data, queries)
    assert out["neighbor_indices"].shape == (5, 2)