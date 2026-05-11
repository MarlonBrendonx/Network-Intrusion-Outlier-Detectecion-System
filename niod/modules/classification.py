"""
Avaliação e busca de hiperparâmetros para classificação supervisionada.

Paralelo ao evaluation.py (anomalia), mas:
- fit(X_train, y_train) com labels 0/1
- predict(X_test) retorna 0/1
- Sem RobustScaler (XGBoost é baseado em árvores, invariante a escala)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Any

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from xgboost import XGBClassifier

from niod.config.settings import CLASSIFICATION_PARAM_GRIDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado da classificação
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    """Resultado estruturado de uma avaliação de classificador."""

    f1: float
    params: dict[str, Any]
    report: str
    confusion_matrix: np.ndarray
    model: Any


# ---------------------------------------------------------------------------
# Factory de modelos supervisionados
# ---------------------------------------------------------------------------
def _create_xgboost(**params: Any) -> XGBClassifier:
    return XGBClassifier(
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
        verbosity=0,
        **params,
    )


CLASSIFICATION_MODEL_REGISTRY: dict[str, Any] = {
    "xgboost": _create_xgboost,
}


def get_classifier_factory(name: str):
    if name not in CLASSIFICATION_MODEL_REGISTRY:
        available = ", ".join(CLASSIFICATION_MODEL_REGISTRY.keys())
        raise KeyError(f"Classificador '{name}' não encontrado. Disponíveis: {available}")
    return CLASSIFICATION_MODEL_REGISTRY[name]


# ---------------------------------------------------------------------------
# Avaliação de modelo individual
# ---------------------------------------------------------------------------
def evaluate_classifier(
    model_factory,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    verbose: bool = True,
) -> ClassificationResult:
    """
    Treina e avalia um classificador supervisionado.

    Args:
        model_factory: Callable que cria o modelo com os parâmetros dados.
        params: Hiperparâmetros do modelo.
        X_train: Dados de treino (ambas as classes).
        y_train: Labels de treino (0=Normal, 1=Ataque).
        X_test: Dados de teste.
        y_test: Labels de teste (0=Normal, 1=Ataque).
        verbose: Se True, loga o F1 durante a avaliação.

    Returns:
        ClassificationResult com métricas e modelo treinado.
    """
    model = model_factory(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, labels=[0, 1], average="macro")

    if verbose:
        logger.info("PARAMS: %s  →  F1: %.4f", params, f1)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal (0)", "Ataque (1)"],
        output_dict=False,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return ClassificationResult(
        f1=f1,
        params=params,
        report=report,
        confusion_matrix=cm,
        model=model,
    )


# ---------------------------------------------------------------------------
# Tqdm + joblib
# ---------------------------------------------------------------------------
import contextlib


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object: tqdm):
    class TqdmCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        tqdm_object.close()


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------
def hyperparameters_search_classifier(
    algorithm_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_jobs: int = 8,
) -> ClassificationResult:
    """
    Busca exaustiva de hiperparâmetros via Grid Search paralelo.

    Avalia todas as combinações do CLASSIFICATION_PARAM_GRID no conjunto
    de validação (nunca no teste).

    Args:
        algorithm_name: Nome do algoritmo (chave em CLASSIFICATION_PARAM_GRIDS).
        X_train: Dados de treino.
        y_train: Labels de treino (0/1).
        X_val: Dados de validação.
        y_val: Labels de validação (0/1).
        n_jobs: Número de jobs paralelos.

    Returns:
        ClassificationResult do melhor modelo encontrado.
    """
    if algorithm_name not in CLASSIFICATION_PARAM_GRIDS:
        raise KeyError(
            f"Grid não definido para '{algorithm_name}'. "
            f"Disponíveis: {list(CLASSIFICATION_PARAM_GRIDS.keys())}"
        )

    param_grid = CLASSIFICATION_PARAM_GRIDS[algorithm_name]
    model_factory = get_classifier_factory(algorithm_name)

    param_list = [
        dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())
    ]

    total = len(param_list)
    logger.info(
        "Grid Search para '%s' — %d combinações de hiperparâmetros.",
        algorithm_name,
        total,
    )

    with _tqdm_joblib(tqdm(desc="Treinando classificadores", total=total)):
        results: list[ClassificationResult] = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_classifier)(
                model_factory,
                params,
                X_train,
                y_train,
                X_val,
                y_val,
                verbose=False,
            )
            for params in param_list
        )

    best = max(results, key=lambda r: r.f1)
    logger.info("Melhor F1 encontrado: %.4f | Params: %s", best.f1, best.params)
    return best
