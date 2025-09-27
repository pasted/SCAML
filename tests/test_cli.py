import subprocess
import sys


def test_cli_train(saved_adata_tmp, tmpdir_out):
    cmd = [
        sys.executable,
        "scripts/scaml_train.py",
        "--adata", str(saved_adata_tmp),
        "--label-key", "label_ref",
        "--split-key", "harvest",
        "--split-train", "H4",
        "--split-test", "H8",
        "--features", "hvg",
        "--n-hvg", "100",
        "--embedding", "X_harmony",
        "--models", "lr",
        "--outdir", str(tmpdir_out),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert (tmpdir_out / "summary.csv").exists()