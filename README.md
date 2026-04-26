# NIOD — Network Intrusion Outlier Detection

Framework profissional para detecção de anomalias em tráfego de rede usando técnicas de **novelty/outlier detection**.

## Arquitetura

```
niod/
├── __init__.py            # Versão do pacote
├── __main__.py            # Entry point: python -m niod
├── main.py                # Orquestrador do pipeline + CLI
├── config/
│   ├── __init__.py
│   └── settings.py        # ExperimentConfig, grids de hiperparâmetros
├── modules/
│   ├── __init__.py
│   ├── models.py          # Registro de modelos (IF, LOF, SVM)
│   ├── evaluation.py      # Treino, avaliação, grid search paralelo
│   └── generalization.py  # Teste de concept drift
├── utils/
│   ├── __init__.py
│   └── data.py            # Loading, limpeza, splits, feature engineering
└── visualization/
    ├── __init__.py
    └── umap_plot.py        # UMAP 3D para análise de drift
```

## Uso

```bash
# Pipeline completo (default: Isolation Forest + novelty + hyper search)
python -m niod

# One-Class SVM sem busca de hiperparâmetros
python -m niod --algorithm svm --no-hyper-search

# LOF com contaminação customizada
python -m niod --algorithm lof --no-novelty --contamination 0.15

# Pular visualização e generalização
python -m niod --skip-umap --skip-generalization

# Debug mode
python -m niod --log-level DEBUG
```

## Algoritmos Suportados

| Algoritmo         | Chave               | Modo Novelty | Modo Outlier |
|-------------------|---------------------|:------------:|:------------:|
| Isolation Forest  | `isolation_forest`  | ✅           | ✅           |
| Local Outlier Factor | `lof`            | ✅           | ✅           |
| One-Class SVM     | `svm`               | ✅           | —            |

## Melhorias sobre a versão original

### Segurança e Integridade dos Dados
- **Sem data leakage**: Imputer fitado SOMENTE no treino, transformado no val/teste
- **Sem variáveis globais mutáveis**: todo estado é encapsulado em dataclasses
- **Sem `import pdb`** em produção
- **Sem código morto**: removidos 400+ linhas de código comentado

### Engenharia de Software
- **Modular**: cada responsabilidade em seu próprio módulo
- **Tipagem**: type hints em todas as funções públicas
- **Dataclasses imutáveis** para configuração (`frozen=True`)
- **Logging estruturado** substituindo `print()` por `logging`
- **CLI completa** com `argparse` e valores default documentados
- **Enum** para algoritmos (evita typos em strings mágicas)
- **Resultados tipados**: `EvaluationResult`, `SplitData`, `GeneralizationResult`

### Reprodutibilidade
- `random_state` propagado consistentemente em todos os splits e modelos
- Configuração centralizada em `ExperimentConfig` (sem valores hardcoded)

### Performance
- Grid search paralelo com `joblib` + `tqdm` (mantido)
- UMAP com `n_jobs=-1` (mantido)
- Figuras fechadas após salvar (`plt.close()`) evitando memory leaks
