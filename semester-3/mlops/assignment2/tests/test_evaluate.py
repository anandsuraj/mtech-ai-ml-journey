from src.evaluate import load_config

def test_load_config():
    cfg = load_config("configs/dev.yaml")
    assert "env" in cfg