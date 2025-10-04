import sys
from pathlib import Path
import pytest

# Locate the package dir that contains "scaml/"
PKG_DIR = Path(__file__).resolve().parents[1]  # either repo_root or SCAML/
if not (PKG_DIR / "scaml").exists():
    PKG_DIR = PKG_DIR / "SCAML"
if (PKG_DIR / "scaml").exists() and str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))


import numpy as np
import pandas as pd
import anndata as ad
import pytest


# Ensure we can import "from scaml import ..." (module lives in SCAML/scaml)
_SCAML_DIR = Path(__file__).resolve().parents[1] / "SCAML"
if str(_SCAML_DIR) not in sys.path:
    sys.path.insert(0, str(_SCAML_DIR))


@pytest.fixture(scope="session")
def toy_h5ad(tmp_path_factory) -> Path:
    """Create a tiny AnnData with integer counts + metadata columns SCAML expects.

    Harvest split:
      - train (H4): 30 cells, labels: {MG1, MG2}; (Monocyte absent in train)
      - test  (H8): 30 cells, labels: {MG1, MG2, Monocyte}
    """
    rng = np.random.default_rng(42)

    n_genes = 50
    n_train = 30
    n_test = 30
    n_cells = n_train + n_test

    genes = [f"G{i}" for i in range(n_genes)]
    cells = [f"C{i:03d}" for i in range(n_cells)]

    # Integer counts (so HVG seurat_v3 won’t complain)
    X = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.int32)

    # Labels (strings)
    labels_train = np.array(["Microglia cluster 1"] * 15 + ["Microglia cluster 2"] * 15)
    labels_test  = np.array(
        ["Microglia cluster 1"] * 10
        + ["Microglia cluster 2"] * 10
        + ["Monocyte"] * 10
    )
    label_ref = np.concatenate([labels_train, labels_test])

    harvest = np.array(["H4"] * n_train + ["H8"] * n_test)
    culture = np.array(["A", "C"] * (n_cells // 2) + (["A"] if n_cells % 2 else []))
    is_hek  = np.array([False] * n_cells)

    obs = pd.DataFrame(
        {
            "label_ref": label_ref,
            "harvest": harvest,
            "culture": culture[:n_cells],
            "is_hek": is_hek,
        },
        index=cells,
    )
    var = pd.DataFrame({"gene": genes}, index=genes)

    # Add a small embedding so --embedding can be tested
    obsm = {"X_harmony": rng.normal(size=(n_cells, 8)).astype(np.float32)}

    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm)
    out = tmp_path_factory.mktemp("adata") / "toy.h5ad"
    adata.write_h5ad(out)
    return out


@pytest.fixture()
def outdir(tmp_path) -> Path:
    return tmp_path / "out"
