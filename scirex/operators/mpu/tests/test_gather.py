import jax
import jax.numpy as jnp
from functools import partial

from scirex.operators.mpu import (
    gather_from_model_parallel_region,
    get_model_parallel_axis,
)

axis = get_model_parallel_axis()


@partial(jax.pmap, axis_name=axis)
def run(x):
    return gather_from_model_parallel_region(x, dim=-1)


def main():

    # each device starts with a local chunk
    x = jnp.arange(8).reshape(jax.device_count(), 4)

    result = run(x)

    print(result)


if __name__ == "__main__":
    main()