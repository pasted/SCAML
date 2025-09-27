def test_subset_split_and_keys(saved_adata_tmp):
    from scaml.io import load_adata, subset_split
    adata = load_adata(saved_adata_tmp)
    train, test = subset_split(adata, "harvest", ["H4"], ["H8"])
    assert train.n_obs > 0 and test.n_obs > 0
    # expected obs keys
    for key in ["harvest", "culture", "label_ref", "is_hek"]:
        assert key in adata.obs.columns


def test_hvg_and_embedding(synthetic_adata):
    from scaml.features import select_hvgs, get_features
    hv_idx, hv_names = select_hvgs(synthetic_adata, n_top=100)
    assert len(hv_names) <= 100 and len(hv_idx) == len(hv_names)
    X_emb = get_features(synthetic_adata, embedding="X_harmony", hv_idx=None)
    assert X_emb.shape[0] == synthetic_adata.n_obs and X_emb.shape[1] == 20