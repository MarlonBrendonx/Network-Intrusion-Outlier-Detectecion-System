# NIOD — Network Intrusion Outlier Detection

Framework for anomaly detection in network traffic using
**novelty/outlier detection** (unsupervised) and **supervised classification** techniques

## Architecture

```
niod/
├── __init__.py            # Package version
├── __main__.py            # Entry point: python -m niod (unsupervised pipeline)
├── main.py                # Outlier detection pipeline orchestrator + CLI
├── classify.py            # Supervised classification pipeline + CLI
├── config/
│   ├── __init__.py
│   └── settings.py        # ExperimentConfig, enums and hyperparameter grids
├── modules/
│   ├── __init__.py
│   ├── models.py          # Unsupervised model registry (IF, LOF, SVM)
│   ├── classification.py  # Supervised classifier registry/training (XGBoost)
│   └── evaluation.py      # Training, evaluation and parallel grid search
├── utils/
│   ├── __init__.py
│   ├── data.py            # Loading, cleaning, statistical filters, splits, PCA reduce
│   └── domain_features.py # Engineered domain features (Eng_* groups)
└── visualization/
    ├── __init__.py
    └── pca_plot.py        # 2D/3D PCA and cross-domain PCA for drift analysis

scripts/
├── check_balancing.py     # SMOTE inspection/balancing of .arff datasets
└── generate_scree_pca.py  # Scree plot + cumulative variance to justify number of components
```

## How to run

### 1. Requirements

- Python **>= 3.10**
- The `.arff` datasets in the `data/` folder (not versioned). By default, the pipeline expects:
  - `data/Friday.arff`
  - `data/Tuesday.arff`


### 2. Installation

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install the package in editable mode
pip install -e .

# (Optional) include support for interactive PCA plots (--pca-cross-interactive)
pip install -e ".[interactive]"
```

Installation registers the `niod` command (equivalent to `python -m niod`) and installs all
dependencies declared in `pyproject.toml` (numpy, pandas, scikit-learn, scipy,
matplotlib, joblib, tqdm, xgboost, imbalanced-learn).

### 3. Unsupervised pipeline

```bash
# Full pipeline (default: Isolation Forest, novelty mode, no hyperparameter search)
python -m niod

# Enable hyperparameter (grid) search
python -m niod --hyper-search

# One-Class SVM
python -m niod --algorithm svm

# LOF in outlier mode (no novelty) with custom contamination
python -m niod --algorithm lof --no-novelty --contamination 0.15

# Add domain features (groups or individual features)
python -m niod --domain-features Eng_Flag_Density Eng_Flow_Rates
python -m niod --all-domain-features

# Train using ONLY a fixed set of columns (bypasses the statistical filters)
python -m niod --feature-whitelist fwd_header_to_payload_ratio "ACK Flag Count"

# Dimensionality reduction via PCA (fitted on training only)
python -m niod --pca-reduce 16

# Generate PCA visualizations (off by default)
python -m niod --plot-pca --plot-pca-cross

# Debug
python -m niod --log-level DEBUG
```

### 4. Supervised pipeline (classification)

```bash
# Classification with XGBoost (no hyperparameter search by default)
python -m niod.classify

# Positional (temporal-proxy) split instead of random — avoids leaking twin flows
# from the same attack episode between train and test
python -m niod.classify --temporal-split

# Train on the real imbalance (disable SMOTE)
python -m niod.classify --no-smote

```

### 5. Helper scripts

```bash
# Inspect a dataset's class balance (and generate a balanced version via SMOTE)
python scripts/check_balancing.py --smote

# Generate PCA scree plot + cumulative variance (saved to docs/)
python scripts/generate_scree_pca.py --train-dataset data/Friday.arff
```

### 6. Outputs

- **Metrics** (F1, classification report, confusion matrix) are printed via `logging`
  to the terminal — adjust verbosity with `--log-level`.
- **PCA figures** (`pca_2d_test_*.png`, `pca_cross_*.png`) are saved to the execution
  directory when `--plot-pca` / `--plot-pca-cross` are enabled. With
  `--pca-cross-interactive`, a rotatable HTML is generated (requires the `interactive` extra).
- The helper scripts' figures (`pca_scree.png`, `pca_cumulative.png`) go to `docs/`.

> The `data/`, `results/`, `*.png` directories and build artifacts are ignored by git.

See all options for each CLI:

```bash
python -m niod --help
python -m niod.classify --help
```

## Supported Algorithms

### Unsupervised

| Algorithm            | Key                | Novelty Mode | Outlier Mode |
|----------------------|--------------------|:------------:|:------------:|
| Isolation Forest     | `isolation_forest` | ✅           | ✅           |
| Local Outlier Factor | `lof`              | ✅           | ✅           |
| One-Class SVM        | `svm`              | ✅           | —           |

### Supervised

| Algorithm | Key       |
|-----------|-----------|
| XGBoost   | `xgboost` |
