"""
Configuration for FNO experiments on Poisson equation (2D and 3D).
"""

class FNO2DConfig:
    # -------------------
    # Model Architecture
    # -------------------
    modes_x = 16
    modes_y = 16
    width = 64
    depth = 4
    input_channels = 1
    output_channels = 1

    # -------------------
    # Training Parameters
    # -------------------
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 32
    epochs = 501  # Total training epochs
    steps_per_epoch = 100  # Number of batches per epoch

    # LR Scheduler (StepLR equivalent)
    scheduler_step_size = 100  # Decay LR every N epochs
    scheduler_gamma = 0.5  # Decay factor

    # -------------------
    # Data Generation
    # -------------------
    nx = 64
    ny = 64
    seed = 42


class FNO3DConfig:
    # -------------------
    # Model Architecture
    # -------------------
    modes_x = 12
    modes_y = 12
    modes_z = 12
    width = 32
    depth = 4
    input_channels = 4  # (source f, x, y, z)
    output_channels = 1

    # -------------------
    # Training Parameters
    # -------------------
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 4  # Reduced for 3D memory constraints
    epochs = 101    # Standard run length
    steps_per_epoch = 50 

    # LR Scheduler (StepLR equivalent)
    scheduler_step_size = 30
    scheduler_gamma = 0.5

    # -------------------
    # Data Generation
    # -------------------
    nx = 32
    ny = 32
    nz = 32
    include_mesh = True
    seed = 42
