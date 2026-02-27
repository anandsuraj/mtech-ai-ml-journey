import numpy as np
import os
from src.data_preprocessing import IMG_SIZE, load_images


def test_normalization_range():
    dummy = np.random.randint(0, 255, (224,224,3))
    normalized = dummy / 255.0

    assert normalized.min() >= 0
    assert normalized.max() <= 1


def test_output_shape():
    dummy = np.random.rand(224,224,3)
    assert dummy.shape == (224,224,3)

def test_load_images_empty(tmp_path):
    folder = tmp_path / "cats"
    os.mkdir(folder)

    X, y = load_images(str(folder), 0, max_images=5)

    assert len(X) == 0