"""
Configurações centralizadas do pipeline NIOD.

Todas as constantes e hiperparâmetros ficam aqui, evitando valores
hardcoded espalhados pelo código. Usa dataclasses para validação
e imutabilidade controlada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
@unique
class Algorithm(str, Enum):
    """Algoritmos de detecção de anomalia suportados."""

    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    SVM = "svm"


# ---------------------------------------------------------------------------
# Configurações do experimento
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    """Parâmetros globais de um experimento de detecção de anomalias."""

    # --- Dados ---
    train_dataset: Path = Path("Friday_balanceado.arff")
    generalization_dataset: Path = Path("data/Tuesday.arff")

    # --- Modo ---
    algorithm: Algorithm = Algorithm.ISOLATION_FOREST
    novelty: bool = True
    hyper_search: bool = True
    contamination: float = 0.1
    apply_filters: bool = True
    pca_reduce: int | None = None
    domain_features: list[str] | None = None

    # --- Split ---
    train_size: float = 0.6
    val_ratio: float = 0.25  # proporção val dentro do restante (1 - train_size)
    random_state: int = 42

    # --- UMAP ---
    umap_n_components: int = 3
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    umap_sample_size: int = 5000

    # --- Logging ---
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination < 1.0:
            raise ValueError(
                f"contamination deve estar em (0, 1), recebido: {self.contamination}"
            )
        if not 0.0 < self.train_size < 1.0:
            raise ValueError(
                f"train_size deve estar em (0, 1), recebido: {self.train_size}"
            )

    @property
    def train_dataset_stem(self) -> str:
        return self.train_dataset.stem

    @property
    def generalization_dataset_stem(self) -> str:
        return self.generalization_dataset.stem


# ---------------------------------------------------------------------------
# Grids de hiperparâmetros
# ---------------------------------------------------------------------------
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    Algorithm.LOF.value: {
        "n_neighbors": [20, 50, 100, 200],
        "metric": ["manhattan", "euclidean"],
        "leaf_size": [30],
    },
    Algorithm.ISOLATION_FOREST.value: {
        # "contamination": [0.05, 0.1, 0.15, 0.2],
        "max_features": [0.5, 0.8, 1.0],
        "max_samples": [1024, 4096, 8192, 16384, "auto"],
        "n_estimators": [100, 200, 300],
        "bootstrap": [False],
    },
    Algorithm.SVM.value: {
        "kernel": ["rbf"],
        "nu": [0.05, 0.10, 0.15, 0.20],
        "gamma": ["scale", 0.1, 0.5, 1.0],
    },
}

# Parâmetros default quando não há busca de hiperparâmetros
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    Algorithm.ISOLATION_FOREST.value: {},
    Algorithm.SVM.value: {},
    Algorithm.LOF.value: {},
}
