"""
Registro de modelos de detecção de anomalia.

Cada factory cria uma instância do modelo com parâmetros validados.
Novos modelos devem ser registrados aqui.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

# Tipo para as factories de modelo
ModelFactory = Callable[..., Any]


def _create_lof(**params: Any) -> LocalOutlierFactor:
    """Factory para Local Outlier Factor com novelty=True por padrão."""
    return LocalOutlierFactor(novelty=True, n_jobs=1, **params)


def _create_isolation_forest(**params: Any) -> IsolationForest:
    """Factory para Isolation Forest com random_state fixo."""
    return IsolationForest(n_jobs=1, random_state=42, **params)


def _create_svm(**params: Any) -> OneClassSVM:
    """Factory para One-Class SVM."""
    return OneClassSVM(**params)


# ---------------------------------------------------------------------------
# Registro central
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, ModelFactory] = {
    "lof": _create_lof,
    "isolation_forest": _create_isolation_forest,
    "svm": _create_svm,
}


def get_model_factory(name: str) -> ModelFactory:
    """
    Retorna a factory de modelo pelo nome.

    Raises:
        KeyError: Se o modelo não está registrado.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Modelo '{name}' não encontrado. Disponíveis: {available}"
        )
    return MODEL_REGISTRY[name]
