def test_train_eval_runs(synthetic_adata, tmpdir_out):
    from scaml.train_eval import train_and_eval
    # split by harvest for domain shift
    tr = synthetic_adata[synthetic_adata.obs["harvest"] == "H4"].copy()
    te = synthetic_adata[synthetic_adata.obs["harvest"] == "H8"].copy()

    train_and_eval(
        tr, te,
        label_key="label_ref",
        models=["lr", "rf"],
        features="hvg",
        n_hvg=100,
        embedding=None,
        outdir=tmpdir_out,
    )
    assert (tmpdir_out / "summary.csv").exists()
    assert any((tmpdir_out / f"report_{m}.csv").exists() for m in ["lr", "rf"]) 