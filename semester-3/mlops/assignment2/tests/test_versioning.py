import os
from src.versioning import get_next_version


def test_next_version_empty(tmp_path):
    v = get_next_version(tmp_path)
    assert v == "v1"


def test_next_version_increment(tmp_path):
    os.mkdir(tmp_path / "v1")
    os.mkdir(tmp_path / "v2")

    v = get_next_version(tmp_path)
    assert v == "v3"