import pytest
from pathlib import Path


def test_novelty_iforest(toy_h5ad, outdir):
    sklearn = pytest.importorskip("sklearn")
    sc = pytest.importorskip("scanpy")

    from scaml.novelty import train_iforest  # expects your novelty module

    adata = sc.read_h5ad(str(toy_h5ad))
    model, scores = train_iforest(
        adata=adata,
        hek_key="is_hek",
        hek_positive=True,
        features="hvg",
        n_hvg=10,
        embedding=None,
        random_state=0,
    )
    assert scores.shape[0] == adata.n_obs
    # sanity: anomaly scores finite
    import numpy as np
    assert np.isfinite(scores).all()
