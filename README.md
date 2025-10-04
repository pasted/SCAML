# SCAML

Machine-learning for single-cell RNA-seq microglia.
Train scikit-learn models (LR / RF / XGB / MLP) to predict cell types/states directly from expression, benchmarked across harvests (H4 vs H8) and with HEK novelty detection. This extends the Seurat analysis from **[scrna_ipsc](https://github.com/pasted/scrna_ipsc)** and turns label transfer into a full ML benchmark.

---

## What’s new (high-level)

* **Robust splits & labels**

  * Train/test by `harvest`, `culture`, etc. with consistent label encoding.
  * Works even when train/test see different class sets (reverse split).
    XGBoost label remapping + probability expansion included.

* **Safer metrics**

  * `classification_report` aligned to the full class list.
  * Macro OVR ROC-AUC computed robustly; returns `NaN` when test has <2 classes.

* **Saved artifacts for plotting**

  * Per-model: `predictions.csv`, `confusion_matrix.csv`, `metrics.json`, `proba.csv`.
  * Top-level: `summary_metrics.json`, `classes.txt`, `features.txt`.
  * New utility: `scaml_plot.py` renders model comparison + confusion matrices.

* **Seurat v5 → AnnData export guidance**

  * Reliable `.h5ad` creation even with Seurat v5 multi-layer assays.

---

## Data expectations (AnnData `.h5ad`)

* `adata.X` — **recommended**: raw counts (SCAML log1p’s internally for HVGs).
  If you prefer, you can skip HVGs and pass an **embedding** via `--embedding`.

* `adata.obs` (columns expected):

  * `label_ref`: string labels (e.g. Olah 2020 microglia clusters).
  * `harvest`: e.g. `"H4"` / `"H8"` (or any split key you choose).
  * `is_hek` (optional): boolean for novelty detection.

* `adata.obsm` (optional but supported):

  * `X_pca`, `X_umap`, and optionally `X_harmony` (if you ran Harmony in Seurat).
  * Use `--embedding X_harmony` (or `X_pca`, etc.) to train on embeddings instead of genes.

---

## Export from Seurat v5 → `.h5ad` (robust path)

Seurat v5’s multi-layer assays can break older converters. A reliable approach is to write the AnnData directly from R using the **`anndata`** R package:

```r
# In R
library(Seurat)
library(anndata)

obj <- readRDS("seurat_with_labels.rds")

# Map fields SCAML expects
stopifnot("olah_label" %in% colnames(obj@meta.data))
obj$label_ref <- as.character(obj$olah_label)
obj$is_hek    <- if ("mg_like" %in% colnames(obj@meta.data)) !obj$mg_like else FALSE

# Choose expression for adata.X (SCT preferred if present; otherwise RNA)
assay <- if ("SCT" %in% names(obj@assays)) "SCT" else "RNA"
DefaultAssay(obj) <- assay
# Ensure 'data' exists; recompute if needed
obj <- NormalizeData(obj, assay = assay, verbose = FALSE)

# Build AnnData
X <- GetAssayData(obj, assay = assay, slot = "data")  # log-normalized
obsm <- list()
if ("pca" %in% names(obj@reductions))     obsm[["X_pca"]]     <- as.matrix(Embeddings(obj, "pca"))
if ("umap" %in% names(obj@reductions))    obsm[["X_umap"]]    <- as.matrix(Embeddings(obj, "umap"))
if ("harmony" %in% names(obj@reductions)) obsm[["X_harmony"]] <- as.matrix(Embeddings(obj, "harmony"))

ad <- anndata::AnnData(
  X   = X,
  obs = obj@meta.data,
  var = data.frame(gene = rownames(obj), row.names = rownames(obj)),
  obsm = obsm
)

dir.create("data", showWarnings = FALSE)
ad$write_h5ad("data/ipsc_mg_all.h5ad", compression = "gzip")
```

**Verify the file** (size >> 1MB and `h5ls` shows groups like `obs`, `var`, `X`, `obsm`).
If you prefer HVGs on raw counts, set `adata.layers["counts"]` too; otherwise SCAML will still work (it log1p’s the selected genes).

> ⚠️ SeuratDisk sometimes creates tiny (invalid) `.h5ad` files under Seurat v5. If your file is ~800 bytes or `h5ls` shows no keys, use the `anndata` route above.

---

## Setup

```bash
# Python (aligns with CI)
conda create -y -n scaml python=3.10
conda activate scaml

# Install
pip install --upgrade pip
pip install -r SCAML/requirements.txt
```

---

## Train & Evaluate

### 1) Baseline cross-harvest split

```bash
python SCAML/scaml_train.py --adata data/ipsc_mg_all.h5ad \
  --label-key label_ref \
  --split-key harvest --split-train H4 --split-test H8 \
  --features hvg --n-hvg 3000 \
  --models lr rf xgb mlp \
  --outdir results/baseline_H4train_H8test
```

### 2) Reverse split (robust to class-set differences)

```bash
python SCAML/scaml_train.py --adata data/ipsc_mg_all.h5ad \
  --label-key label_ref \
  --split-key harvest --split-train H8 --split-test H4 \
  --features hvg --n-hvg 3000 \
  --models lr rf xgb mlp \
  --outdir results/baseline_H8train_H4test
```

### 3) Learning curves (group-aware; avoids leakage)

```bash
python SCAML/scaml_learning_curve.py \
  --adata data/ipsc_mg_all.h5ad \
  --label-key label_ref --group-key culture \
  --features hvg --n-hvg 3000 \
  --model lr \
  --outdir results/learning_curves
```

### 4) Novelty detection (HEK outliers)

```bash
python SCAML/scaml_novelty.py \
  --adata data/ipsc_mg_all.h5ad \
  --hek-key is_hek --hek-positive true \
  --features hvg --n-hvg 3000 \
  --method iforest \
  --outdir results/novelty
```

### 5) Plots

```bash
# Compare models + per-model confusion matrices
python SCAML/scaml_plot.py \
  --results results/baseline_H4train_H8test \
  --outdir  results/baseline_H4train_H8test/plots \
  --models  lr rf xgb mlp

python SCAML/scaml_plot.py \
  --results results/baseline_H8train_H4test \
  --outdir  results/baseline_H8train_H4test/plots \
  --models  lr rf xgb mlp
```

**Tip:** If you exported Harmony to `obsm['X_harmony']`, you can train on it:

```bash
python SCAML/scaml_train.py \
  --adata data/ipsc_mg_all.h5ad \
  --label-key label_ref \
  --split-key harvest --split-train H4 --split-test H8 \
  --embedding X_harmony \
  --models lr rf xgb mlp \
  --outdir results/with_harmony_H4train_H8test
```

---

## Outputs

**Top-level (per run)**

* `summary_metrics.json` — per-model summary: accuracy, F1 (macro), ROC-AUC (macro OVR; may be `NaN` if test has one class)
* `classes.txt` — integer ↔︎ label mapping
* `features.txt` — HVG list or embedding column names

**Per model (e.g., `lr/`, `rf/`, `xgb/`, `mlp/`)**

* `classification_report.txt` — precision/recall/F1 by class
* `confusion_matrix.csv` — rows = true, cols = predicted
* `predictions.csv` — cell, true/pred labels + indices
* `proba.csv` — per-class probabilities (expanded to the full class set)
* `metrics.json` — accuracy, F1 (macro), ROC-AUC (macro OVR)

**Plots (from `scaml_plot.py`)**

* `plots/model_compare.png` — accuracy & F1 bars per model
* `plots/<model>_confusion_matrix.png` — heatmaps (row-normalized)

---

## Models

* **LR** — `LogisticRegression` (with `MaxAbsScaler`)
* **RF** — `RandomForestClassifier`
* **XGB** — `XGBClassifier` (auto-remaps labels when train/test class sets differ)
* **MLP** — `MLPClassifier` (dense path with standardization)

---

## Notes & gotchas

* **HVGs vs counts**: The HVG selector is Seurat-style. For cleanest results, put **raw counts** in `adata.X`; SCAML applies `log1p` on the selected genes. If your `adata.X` is already log data, consider using `--embedding` to avoid HVG warnings.

* **ROC-AUC showing `NaN`**: This is expected if your test split contains only one class. Accuracy/F1 are still valid.

* **SeuratDisk tiny `.h5ad`**: If `h5ad` is ~800 bytes or missing groups (`obs`, `var`, etc.), re-export with the R `anndata` approach above.

* **Group-aware splits**: Prefer `--split-key culture` (or donor/patient) for fair generalization tests; avoid random cell splits.

---

## Dev / CI

Local:

```bash
pip3 install flake8 pytest
flake8 .
pytest
```

GitHub Actions (name: **SCAML**) runs:

* Lint: `flake8` (syntax/complexity)
* Tests: `pytest` on Python 3.10

---

## Acknowledgements

* Built on top of your Seurat v5 pipeline (SCT, optional Harmony, label transfer to Olah 2020).
* Uses: Scanpy/AnnData, scikit-learn, (optional) XGBoost, and R packages for export.
