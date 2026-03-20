import jax
import jax.numpy as jnp
from scirex.operators.layers.gno_block import GNOBlock

def test_gno_block_jit():
    print("Testing GNOBlock JIT...")
    block = GNOBlock(
        in_channels=3,
        out_channels=8,
        coord_dim=2,
        radius=1.5,
        max_neighbors=4,
        transform_type="linear",
        reduction="mean"  # Test with segment_mean
    )

    y = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    x = jnp.array([[0.5, 0.5], [1.5, 1.5]])
    f_y = jnp.ones((4, 3))
    
    rng = jax.random.PRNGKey(0)
    
    # Init
    variables = block.init(rng, y=y, x=x, f_y=f_y)
    
    @jax.jit
    def apply_fn(y, x, f_y):
        return block.apply(variables, y=y, x=x, f_y=f_y)
    
    # Forward JIT
    out = apply_fn(y=y, x=x, f_y=f_y)
    
    print("Output shape unbatched JIT:", out.shape)
    assert out.shape == (2, 8)

if __name__ == "__main__":
    test_gno_block_jit()
    print("Success JIT!")
