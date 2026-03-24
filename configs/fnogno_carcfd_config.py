from dataclasses import dataclass, field
from typing import Optional, Literal, List
from configs.models import FNOGNO_Small3d


@dataclass
class FNOGNOCarCFDConfig:
    """Full experiment config for Car-CFD + FNOGNO."""

    # ── Model Architecture (preset) ──
    model: FNOGNO_Small3d = field(default_factory=FNOGNO_Small3d)

    # ── Training Parameters ──
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 100
    seed: int = 42

    # ── LR Scheduler ──
    scheduler_type: Literal["step", "cosine"] = "cosine"
    cosine_decay_epochs: int = 100
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5

    # ── Data Parameters ──
    data_root: str = "./scirex/operators/data/car_cfd_data"
    query_res: list = field(default_factory=lambda: [32, 32, 32])
    n_train: int = 100
    n_test: int = 20
    download: bool = True

    # ── Neighbor Search Cache ──
    neighbor_cache_dir: str = "./scirex/operators/data/neighbor_cache"

    # ── Logging ──
    wandb_log: bool = False

    # ── Convenience properties ──
    @property
    def fno_n_modes(self):
        return self.model.fno_n_modes

    @property
    def fno_hidden_channels(self):
        return self.model.fno_hidden_channels

    @property
    def fno_n_layers(self):
        return self.model.fno_n_layers

    @property
    def fno_lifting_channel_ratio(self):
        return self.model.fno_lifting_channel_ratio

    @property
    def gno_radius(self):
        return self.model.gno_radius

    @property
    def in_channels(self):
        return self.model.in_channels

    @property
    def out_channels(self):
        return self.model.out_channels

    @property
    def gno_coord_dim(self):
        return self.model.gno_coord_dim

    @property
    def gno_transform_type(self):
        return self.model.gno_transform_type

    @property
    def max_neighbors(self):
        return self.model.max_neighbors

    @property
    def use_neighbor_cache(self):
        return self.model.use_neighbor_cache
