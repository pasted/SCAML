def test_novelty_iforest(synthetic_adata, tmpdir_out):
    from scaml.novelty import novelty_eval
    novelty_eval(
        synthetic_adata,
        hek_key="is_hek",
        hek_positive=True,
        features="hvg",
        n_hvg=100,
        embedding="X_harmony",
        outdir=tmpdir_out,
        method="iforest",
    )
    metrics = (tmpdir_out / "metrics.txt").read_text()
    assert "AUROC" in metrics