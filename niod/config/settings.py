from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@unique
class Algorithm(str, Enum):
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    SVM = "svm"


@unique
class ClassificationAlgorithm(str, Enum):
    XGBOOST = "xgboost"


@dataclass(frozen=True)
class ExperimentConfig:
    train_dataset: Path = Path("Friday.arff")
    generalization_dataset: Path = Path("data/Tuesday.arff")
    extra_normal_dataset: Path | None = None
    few_shot_dataset: Path | None = None
    few_shot_ratio: float = 0.05

    algorithm: Algorithm = Algorithm.ISOLATION_FOREST
    novelty: bool = True
    hyper_search: bool = False
    contamination: float = 0.1
    apply_filters: bool = True
    pca_reduce: int | None = None
    domain_features: list[str] | None = None
    feature_whitelist: list[str] | None = None

    train_size: float = 0.6
    val_ratio: float = 0.25
    random_state: int = 42

    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination < 1.0:
            raise ValueError(
                f"contamination must be in (0, 1), got: {self.contamination}"
            )
        if not 0.0 < self.train_size < 1.0:
            raise ValueError(
                f"train_size must be in (0, 1), got: {self.train_size}"
            )

    @property
    def train_dataset_stem(self) -> str:
        return self.train_dataset.stem

    @property
    def generalization_dataset_stem(self) -> str:
        return self.generalization_dataset.stem


PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    Algorithm.LOF.value: {
        "n_neighbors": [10, 20, 35, 50, 100],
        "metric": ["manhattan", "euclidean"],
        "contamination": [
            0.05,
            0.1,
            0.15,
        ],
    },
    Algorithm.ISOLATION_FOREST.value: {
        "contamination": [0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        "max_features": [0.5, 0.8, 1.0],
        "max_samples": [256, 512, 1024],
        "n_estimators": [100, 200, 300],
        "bootstrap": [False],
    },
    Algorithm.SVM.value: {
        "kernel": ["rbf"],
        "nu": [0.05, 0.10, 0.15, 0.20],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },
}

CLASSIFICATION_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    ClassificationAlgorithm.XGBOOST.value: {
        "n_estimators": [200, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    },
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    Algorithm.ISOLATION_FOREST.value: {},
    Algorithm.SVM.value: {},
    Algorithm.LOF.value: {},
}
