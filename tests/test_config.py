from configs.config_loader import load_config, Config

def test_load_default_config():
    config = load_config("configs/default.yaml")
    assert isinstance(config, Config)
    assert config.system.seed == 42
    assert config.training.epochs == 100
    assert config.models.vae.latent_channels == 4

def test_load_debug_config():
    config = load_config("configs/debug.yaml")
    assert isinstance(config, Config)
    assert config.system.seed == 42
    assert config.data.resolution == [64, 64]
    assert config.data.num_frames == 4
    assert config.training.epochs == 2

def test_config_overrides():
    overrides = {
        "system": {"seed": 999},
        "training": {"epochs": 200}
    }
    config = load_config("configs/default.yaml", overrides=overrides)
    assert config.system.seed == 999
    assert config.training.epochs == 200
    assert config.models.vae.latent_channels == 4
