from pathlib import Path
import src.promote as promote


def test_promote_runs(tmp_path, monkeypatch):

    # -----------------------------
    # Fake folder structure
    # -----------------------------
    dev_v1 = tmp_path / "models" / "dev" / "v1"
    prod_dir = tmp_path / "models" / "prod"

    dev_v1.mkdir(parents=True)
    prod_dir.mkdir(parents=True)

    (dev_v1 / "model.h5").write_text("dummy")

    # -----------------------------
    # Monkeypatch version functions
    # -----------------------------
    monkeypatch.setattr(
        promote,
        "get_latest_version",
        lambda x: "v1"
    )

    monkeypatch.setattr(
        promote,
        "get_next_version",
        lambda x: "v1"
    )

    # -----------------------------
    # Monkeypatch os paths
    # -----------------------------
    monkeypatch.setattr(
        promote,
        "os",
        __import__("os")
    )

    # force working directory to tmp
    monkeypatch.chdir(tmp_path)

    # -----------------------------
    # Run promote
    # -----------------------------
    promote.promote()

    # -----------------------------
    # Verify promotion happened
    # -----------------------------
    assert (tmp_path / "models" / "prod" / "v1").exists()