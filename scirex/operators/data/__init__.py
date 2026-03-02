# scirex/operators/data/__init__.py
from .poisson import random_poisson_2d_batch, random_poisson_3d_batch, generator_2d, generator_3d
from .darcy import random_darcy_batch, generator as darcy_generator
from .darcy_zenodo import load_darcy_numpy, generator_from_numpy
