# SCAML
Single Cell RNA analysis pipeline for Seurat outputs of microglial iPSC, using Random Forests.

This project is an extension of the microglial iPSC Seurat classification and analysis. The input pipeline starts with https://github.com/pasted/scrna_ipsc

The pipeline extends the previous analysis and benchmark machine‑learning models for cell type/state prediction on your iPSC‑microglia scRNA‑seq datasets (H4 vs H8 harvests; HEK controls), comparing against reference label transfer, and testing robustness to chip/harvest effects. Includes novelty detection for HEK controls.

Scikit Learn models included are:

* LR - LogisticRegression
* RF - RandomForestClassifier
* XGB - XGBClassifier
* MLP - MLPClassifier (Multi-layer Perceptron classifier)


## Setup


```bash

conda create -y -n scaml python=3.11
conda activate scaml
pip install -r requirements.txt

# 1) Train & evaluate (baseline models, cross‑harvest splits)
python scripts/scaml_train.py \
--adata data/ipsc_mg_all.h5ad \
--label-key label_ref \
--split-key harvest \
--split-train H4 --split-test H8 \
--features hvg --n-hvg 3000 --embedding X_harmony \
--models lr rf xgb mlp \
--outdir results/baseline_H4train_H8test


# 2) Reverse split
python scripts/scaml_train.py --adata data/ipsc_mg_all.h5ad --label-key label_ref \
--split-key harvest --split-train H8 --split-test H4 \
--features hvg --n-hvg 3000 --embedding X_harmony \
--models lr rf xgb mlp \
--outdir results/baseline_H8train_H4test


# 3) Learning curves (by cultures/groups to avoid leakage)
python scripts/scaml_learning_curve.py \
--adata data/ipsc_mg_all.h5ad \
--label-key label_ref --group-key culture \
--embedding X_harmony --features hvg --n-hvg 3000 \
--model lr \
--outdir results/learning_curves


# 4) Novelty detection for HEK controls
python scripts/scaml_novelty.py \
--adata data/ipsc_mg_all.h5ad \
--hek-key is_hek --hek-positive true \
--embedding X_harmony --features hvg --n-hvg 3000 \
--method iforest \
--outdir results/novelty
```


### Outputs
- Metrics: accuracy, macro/micro‑F1, AUROC per class, confusion matrices (png), classification reports (csv).
- Plots: UMAP colored by ML predictions vs Seurat clusters/labels; learning curves; SHAP importances for top models.
- Artifacts: trained model pickles (`.joblib`), feature sets and scalers, selected HVGs, embeddings used.


### Notes
- Harmony: compute in R (Seurat + Harmony) and store as `obsm['X_harmony']` in `.h5ad`. SCAML can then consume that embedding directly.
- Group‑aware splits: avoid cell‑level leakage by splitting on `culture` or patient source.
- HEK novelty: train one‑class model on microglia only; score HEK as outliers.