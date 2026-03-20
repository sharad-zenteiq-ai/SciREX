import jax
import jax.numpy as jnp
from scirex.operators.layers.gno_block import GNOBlock

def test_gno_block():
    print("Testing GNOBlock...")
    block = GNOBlock(
        in_channels=3,
        out_channels=8,
        coord_dim=2,
        radius=1.5,
        max_neighbors=4,
        transform_type="linear"
    )

    y = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    x = jnp.array([[0.5, 0.5], [1.5, 1.5]])
    f_y = jnp.ones((4, 3))
    
    rng = jax.random.PRNGKey(0)
    
    # Init
    variables = block.init(rng, y=y, x=x, f_y=f_y)
    
    # Forward
    out = block.apply(variables, y=y, x=x, f_y=f_y)
    
    print("Output shape unbatched:", out.shape)
    
    # Batched Forward
    f_y_batch = jnp.ones((2, 4, 3))
    out_batch = block.apply(variables, y=y, x=x, f_y=f_y_batch)
    print("Output shape batched:", out_batch.shape)

    assert out.shape == (2, 8)
    assert out_batch.shape == (2, 2, 8)

if __name__ == "__main__":
    test_gno_block()
    print("Success!")
