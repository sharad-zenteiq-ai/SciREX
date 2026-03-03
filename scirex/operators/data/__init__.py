# scirex/operators/data/__init__.py
from .poisson import random_poisson_2d_batch, random_poisson_3d_batch, generator_2d, generator_3d, solve_poisson_periodic_batch_2d

# Aliases for backward compatibility
random_poisson_batch = random_poisson_2d_batch
generator = generator_2d
solve_poisson_periodic_batch = solve_poisson_periodic_batch_2d
