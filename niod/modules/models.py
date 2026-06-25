from __future__ import annotations

import logging
from typing import Any, Callable

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

ModelFactory = Callable[..., Any]


def _create_lof(**params: Any) -> LocalOutlierFactor:
    return LocalOutlierFactor(novelty=True, n_jobs=1, **params)


def _create_isolation_forest(**params: Any) -> IsolationForest:
    return IsolationForest(n_jobs=1, random_state=42, **params)


def _create_svm(**params: Any) -> OneClassSVM:
    return OneClassSVM(**params)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "lof": _create_lof,
    "isolation_forest": _create_isolation_forest,
    "svm": _create_svm,
}


def get_model_factory(name: str) -> ModelFactory:
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return MODEL_REGISTRY[name]
