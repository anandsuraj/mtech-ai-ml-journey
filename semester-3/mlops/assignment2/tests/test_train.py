from src.train import build_model


def test_build_model():
    model = build_model()

    assert model is not None
    assert len(model.layers) > 0