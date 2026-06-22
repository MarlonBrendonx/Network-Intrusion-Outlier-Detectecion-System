# NIOD — Network Intrusion Outlier Detection

Framework para detecção de anomalias em tráfego de rede usando técnicas de
**novelty/outlier detection** (não supervisionado) e **classificação supervisionada**,
com foco em avaliação de generalização sob *concept drift* (CICIDS Friday → Tuesday).

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

## Uso

### Pipeline não supervisionado (outlier/novelty detection)

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

### Pipeline supervisionado (classificação)

```bash
# Classificação com XGBoost + grid search
python -m niod.classify

# Few-shot supervisionado: injeta uma fração dos ataques do alvo no treino
python -m niod.classify --few-shot-dataset data/Tuesday.arff --few-shot-ratio 0.05

# Com features de domínio e sem busca de hiperparâmetros
python -m niod.classify --all-domain-features --no-hyper-search
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

## Princípios de engenharia

### Integridade dos dados
- **Sem data leakage**: scaler/imputer/PCA fitados SOMENTE no treino e aplicados a val/teste/generalização.
- **Estado encapsulado** em dataclasses (`ExperimentConfig`, `EvaluationResult`, `SplitData`).

### Reprodutibilidade
- `random_state` propagado consistentemente em splits e modelos.
- Configuração centralizada em `ExperimentConfig` (sem valores hardcoded).

### Performance
- Grid search paralelo com `joblib` + `tqdm`.
- Figuras fechadas após salvar (`plt.close()`) evitando memory leaks.
