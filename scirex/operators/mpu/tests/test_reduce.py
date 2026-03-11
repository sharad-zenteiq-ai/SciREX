import jax
import jax.numpy as jnp
from functools import partial

from scirex.operators.mpu import (
    reduce_from_model_parallel_region,
    get_model_parallel_axis,
)

axis = get_model_parallel_axis()


@partial(jax.pmap, axis_name=axis)
def run(x):
    return reduce_from_model_parallel_region(x)


def main():

    x = jnp.arange(jax.device_count())

    result = run(x)

    print(result)


if __name__ == "__main__":
    main()