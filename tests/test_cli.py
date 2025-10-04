import json
import subprocess
import sys
from pathlib import Path
import pytest


def _find_train_script(start: Path) -> Path:
    # Try typical locations
    candidates = [
        start / "SCAML" / "scaml_train.py",        # repo_root/SCAML/scaml_train.py
        start.parent / "scaml_train.py",           # SCAML/scaml_train.py (if tests live in SCAML/tests)
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: glob from repo root
    for c in (start).rglob("scaml_train.py"):
        if (c.parent.name == "SCAML") and c.is_file():
            return c
    return Path("MISSING")


def test_cli_train(toy_h5ad, outdir):
    pytest.importorskip("scanpy")
    repo_root = Path(__file__).resolve().parents[2]  # repo root when tests are in SCAML/tests
    script = _find_train_script(repo_root)
    assert script.exists(), f"Missing script: {script}"

    cmd = [
        sys.executable, str(script),
        "--adata", str(toy_h5ad),
        "--label-key", "label_ref",
        "--split-key", "harvest",
        "--split-train", "H4",
        "--split-test", "H8",
        "--features", "hvg",
        "--n-hvg", "10",
        "--models", "lr",
        "--outdir", str(outdir),
    ]
    subprocess.run(cmd, check=True)

    summary = outdir / "summary_metrics.json"
    assert summary.exists()
    with summary.open() as fh:
        data = json.load(fh)
    assert "lr" in data and "accuracy" in data["lr"]

