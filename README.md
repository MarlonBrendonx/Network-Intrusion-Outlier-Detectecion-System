# NIOD — Network Intrusion Outlier Detection

Framework para detecção de anomalias em tráfego de rede usando técnicas de
**novelty/outlier detection** (não supervisionado) e **classificação supervisionada**,

## Arquitetura

```
niod/
├── __init__.py            # Versão do pacote
├── __main__.py            # Entry point: python -m niod (pipeline não supervisionado)
├── main.py                # Orquestrador do pipeline de outlier detection + CLI
├── classify.py            # Pipeline de classificação supervisionada + CLI
├── config/
│   ├── __init__.py
│   └── settings.py        # ExperimentConfig, enums e grids de hiperparâmetros
├── modules/
│   ├── __init__.py
│   ├── models.py          # Registro de modelos não supervisionados (IF, LOF, SVM)
│   ├── classification.py  # Registro/treino de classificadores supervisionados (XGBoost)
│   └── evaluation.py      # Treino, avaliação e grid search paralelo
├── utils/
│   ├── __init__.py
│   ├── data.py            # Loading, limpeza, filtros estatísticos, splits, PCA reduce
│   └── domain_features.py # Features de domínio engenheiradas (grupos Eng_*)
└── visualization/
    ├── __init__.py
    └── pca_plot.py        # PCA 2D/3D e PCA cross-domain para análise de drift

scripts/
├── check_balancing.py     # Inspeção/balanceamento via SMOTE dos datasets .arff
└── generate_scree_pca.py  # Scree plot + variância acumulada para justificar nº de componentes
```

## Como executar

### 1. Requisitos

- Python **>= 3.10**
- Os datasets `.arff` na pasta `data/` (não versionada). O pipeline espera, por padrão:
  - `data/Friday_balanceado.arff` — dataset de treino
  - `data/Tuesday.arff` — dataset alvo para avaliação de generalização / few-shot

### 2. Instalação

```bash
# Criar e ativar o ambiente virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Instalar o pacote em modo editável
pip install -e .

# (Opcional) incluir suporte a gráficos PCA interativos (--pca-cross-interactive)
pip install -e ".[interactive]"
```

A instalação registra o comando `niod` (equivalente a `python -m niod`) e instala todas
as dependências declaradas no `pyproject.toml` (numpy, pandas, scikit-learn, scipy,
matplotlib, joblib, tqdm, xgboost, imbalanced-learn).

### 3. Pipeline não supervisionado (outlier/novelty detection)

```bash
# Pipeline completo (default: Isolation Forest + novelty + grid search)
python -m niod

# One-Class SVM sem busca de hiperparâmetros
python -m niod --algorithm svm --no-hyper-search

# LOF em modo outlier (sem novelty) com contaminação customizada
python -m niod --algorithm lof --no-novelty --contamination 0.15

# Avaliar generalização com few-shot não supervisionado (5% das normais do alvo no treino)
python -m niod --few-shot-dataset data/Tuesday.arff --few-shot-ratio 0.05

# Adicionar features de domínio (grupos ou features individuais)
python -m niod --domain-features Eng_Flag_Density Eng_Flow_Rates
python -m niod --all-domain-features

# Redução de dimensionalidade via PCA (fitado só no treino)
python -m niod --pca-reduce 16

# Pular visualizações
python -m niod --skip-pca --skip-pca-cross

# Debug
python -m niod --log-level DEBUG
```

### 4. Pipeline supervisionado (classificação)

```bash
# Classificação com XGBoost + grid search
python -m niod.classify

# Few-shot supervisionado: injeta uma fração dos ataques do alvo no treino
python -m niod.classify --few-shot-dataset data/Tuesday.arff --few-shot-ratio 0.05

# Com features de domínio e sem busca de hiperparâmetros
python -m niod.classify --all-domain-features --no-hyper-search
```

### 5. Scripts auxiliares

```bash
# Inspecionar o balanço de classes de um dataset (e gerar versão balanceada via SMOTE)
python scripts/check_balancing.py --smote

# Gerar scree plot + variância acumulada do PCA (salvos em docs/)
python scripts/generate_scree_pca.py --train-dataset data/Friday_balanceado.arff
```

### 6. Saídas

- **Métricas** (F1, classification report, matriz de confusão) são impressas via `logging`
  no terminal — ajuste a verbosidade com `--log-level`.
- **Figuras PCA** (`pca_2d_test_*.png`, `pca_cross_*.png`) são salvas no diretório de
  execução. Com `--pca-cross-interactive` é gerado um HTML rotacionável (requer o extra
  `interactive`).
- As figuras dos scripts auxiliares (`pca_scree.png`, `pca_cumulativa.png`) vão para `docs/`.

> Diretórios `data/`, `results/`, `*.png` e artefatos de build são ignorados pelo git.

Ver todas as opções de cada CLI:

```bash
python -m niod --help
python -m niod.classify --help
```

## Algoritmos Suportados

### Não supervisionados

| Algoritmo            | Chave              | Modo Novelty | Modo Outlier |
|----------------------|--------------------|:------------:|:------------:|
| Isolation Forest     | `isolation_forest` | ✅           | ✅           |
| Local Outlier Factor | `lof`              | ✅           | ✅           |
| One-Class SVM        | `svm`              | ✅           | —            |

### Supervisionados

| Algoritmo | Chave     |
|-----------|-----------|
| XGBoost   | `xgboost` |

## Features de domínio

Grupos engenheirados disponíveis (via `--domain-features <GRUPO>` ou `--all-domain-features`):

`Eng_Packet_Shape`, `Eng_Fwd_Header_Load`, `Eng_Temporal_Burstiness`,
`Eng_Flag_Density`, `Eng_Flow_Indicators`, `Eng_Flow_Rates`.

Também é possível ativar features individuais (ex.: `--domain-features is_short_flow is_unidirectional`).
