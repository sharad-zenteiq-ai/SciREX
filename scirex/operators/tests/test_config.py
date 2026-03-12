"""
Test that config presets instantiate with expected defaults.
"""
from configs.models import FNOConfig, FNO_Medium2D, FNO_Medium3D

def test_fno_config_defaults():
    config = FNOConfig()
    assert config.arch == "fno"
    assert config.hidden_channels == 64
    assert config.n_layers == 4
    assert config.use_grid is True

def test_fno_medium_2d_preset():
    config = FNO_Medium2D()
    assert config.n_modes == (24, 24)
    assert config.hidden_channels == 128
    assert config.use_norm is True

def test_fno_medium_3d_preset():
    config = FNO_Medium3D()
    assert config.n_modes == (16, 16, 16)
    assert config.hidden_channels == 128
    assert config.use_norm is True
