# NIOD — Network Intrusion Outlier Detection

<<<<<<< Updated upstream
Framework para detecção de anomalias em tráfego de rede usando técnicas de
**novelty/outlier detection** (não supervisionado) e **classificação supervisionada**,
=======
Framework for anomaly detection in network traffic using
**novelty/outlier detection** (unsupervised) and **supervised classification** techniques,
focused on evaluating generalization under *concept drift* (CICIDS Friday → Tuesday).
>>>>>>> Stashed changes

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
  - `data/Friday_balanceado.arff` — training dataset
  - `data/Tuesday.arff` — target dataset for generalization / few-shot evaluation

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

### 3. Unsupervised pipeline (outlier/novelty detection)

```bash
# Full pipeline (default: Isolation Forest + novelty + grid search)
python -m niod

# One-Class SVM without hyperparameter search
python -m niod --algorithm svm --no-hyper-search

# LOF in outlier mode (no novelty) with custom contamination
python -m niod --algorithm lof --no-novelty --contamination 0.15

# Evaluate generalization with unsupervised few-shot (5% of the target's normals in training)
python -m niod --few-shot-dataset data/Tuesday.arff --few-shot-ratio 0.05

# Add domain features (groups or individual features)
python -m niod --domain-features Eng_Flag_Density Eng_Flow_Rates
python -m niod --all-domain-features

# Dimensionality reduction via PCA (fitted on training only)
python -m niod --pca-reduce 16

# Skip visualizations
python -m niod --skip-pca --skip-pca-cross

# Debug
python -m niod --log-level DEBUG
```

### 4. Supervised pipeline (classification)

```bash
# Classification with XGBoost + grid search
python -m niod.classify

# Supervised few-shot: injects a fraction of the target's attacks into training
python -m niod.classify --few-shot-dataset data/Tuesday.arff --few-shot-ratio 0.05

# With domain features and without hyperparameter search
python -m niod.classify --all-domain-features --no-hyper-search
```

### 5. Helper scripts

```bash
# Inspect a dataset's class balance (and generate a balanced version via SMOTE)
python scripts/check_balancing.py --smote

# Generate PCA scree plot + cumulative variance (saved to docs/)
python scripts/generate_scree_pca.py --train-dataset data/Friday_balanceado.arff
```

### 6. Outputs

- **Metrics** (F1, classification report, confusion matrix) are printed via `logging`
  to the terminal — adjust verbosity with `--log-level`.
- **PCA figures** (`pca_2d_test_*.png`, `pca_cross_*.png`) are saved to the execution
  directory. With `--pca-cross-interactive`, a rotatable HTML is generated (requires the
  `interactive` extra).
- The helper scripts' figures (`pca_scree.png`, `pca_cumulativa.png`) go to `docs/`.

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

## Domain features

Available engineered groups (via `--domain-features <GROUP>` or `--all-domain-features`):

`Eng_Packet_Shape`, `Eng_Fwd_Header_Load`, `Eng_Temporal_Burstiness`,
`Eng_Flag_Density`, `Eng_Flow_Indicators`, `Eng_Flow_Rates`.

<<<<<<< Updated upstream
Também é possível ativar features individuais (ex.: `--domain-features is_short_flow is_unidirectional`).
=======
You can also enable individual features (e.g.: `--domain-features is_short_flow is_unidirectional`).

## Engineering principles

### Data integrity
- **No data leakage**: scaler/imputer/PCA fitted ONLY on training and applied to validation/test/generalization.
- **Encapsulated state** in dataclasses (`ExperimentConfig`, `EvaluationResult`, `SplitData`).

### Reproducibility
- `random_state` propagated consistently across splits and models.
- Centralized configuration in `ExperimentConfig` (no hardcoded values).

### Performance
- Parallel grid search with `joblib` + `tqdm`.
- Figures closed after saving (`plt.close()`) to avoid memory leaks.
>>>>>>> Stashed changes
