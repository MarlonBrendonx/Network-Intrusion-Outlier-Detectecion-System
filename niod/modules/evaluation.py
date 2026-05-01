"""
Avaliação de modelos e busca de hiperparâmetros.

Responsabilidades:
- Treinar e avaliar um modelo com métricas padrão.
- Grid search paralelo com progress bar.
- Retornar resultados estruturados (sem side effects).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from itertools import product
from typing import Any

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from niod.config.settings import PARAM_GRIDS
from niod.modules.models import ModelFactory, get_model_factory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado da avaliação
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Resultado estruturado de uma avaliação de modelo."""

    f1: float
    params: dict[str, Any]
    report: str
    confusion_matrix: np.ndarray
    pipeline: Pipeline


# ---------------------------------------------------------------------------
# Avaliação de modelo individual
# ---------------------------------------------------------------------------
def evaluate_model(
    model_factory: ModelFactory,
    params: dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test_transformed: np.ndarray,
    *,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Treina e avalia um modelo dentro de um Pipeline (Scaler → Modelo).

    O Pipeline garante que o scaler é fitado apenas no treino e aplicado
    consistentemente no teste (sem data leakage).

    Args:
        model_factory: Callable que cria o modelo com os parâmetros dados.
        params: Hiperparâmetros do modelo.
        X_train: Dados de treino (já imputados).
        X_test: Dados de teste (já imputados).
        y_test_transformed: Labels do teste no formato 1/-1.
        verbose: Se True, imprime o F1 durante a avaliação.

    Returns:
        EvaluationResult com métricas e pipeline treinado.
    """
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("estimator", model_factory(**params)),
        ]
    )

    estimator = pipeline.named_steps["estimator"]
    is_lof = estimator.__class__.__name__ == "LocalOutlierFactor"
    is_novelty = getattr(estimator, "novelty", False)

    if is_lof and not is_novelty:
        # LOF padrão: fit_predict no teste (ignora treino)
        logger.debug("[LOF] Modo Outlier Detection (fit_predict no teste)")
        y_pred = pipeline.fit_predict(X_test)
    else:
        pipeline.fit(X_train)
        y_pred = pipeline.predict(X_test)

    f1 = f1_score(
        y_test_transformed,
        y_pred,
        labels=[1, -1],
        average="macro",
    )

    if verbose:
        logger.info("PARAMS: %s  →  F1: %.4f", params, f1)

    report = classification_report(
        y_test_transformed,
        y_pred,
        labels=[1, -1],
        target_names=["Normal (1)", "Outlier (-1)"],
        output_dict=False,
    )

    cm = confusion_matrix(y_test_transformed, y_pred, labels=[1, -1])

    return EvaluationResult(
        f1=f1,
        params=params,
        report=report,
        confusion_matrix=cm,
        pipeline=pipeline,
    )


# ---------------------------------------------------------------------------
# Tqdm + joblib integration
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _tqdm_joblib(tqdm_object: tqdm):
    """Context manager para integrar tqdm com joblib Parallel."""

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
def hyperparameters_search(
    algorithm_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val_transformed: np.ndarray,
    *,
    n_jobs: int = 10,
) -> EvaluationResult:
    """
    Busca exaustiva de hiperparâmetros via Grid Search paralelo.

    Avalia todas as combinações do PARAM_GRID no conjunto de VALIDAÇÃO
    (nunca no teste), retornando o melhor resultado.

    Args:
        algorithm_name: Nome do algoritmo (chave em PARAM_GRIDS).
        X_train: Dados de treino.
        X_val: Dados de validação.
        y_val_transformed: Labels de validação (formato 1/-1).
        n_jobs: Número de jobs paralelos (-1 = todos os cores).

    Returns:
        EvaluationResult do melhor modelo encontrado.
    """
    if algorithm_name not in PARAM_GRIDS:
        raise KeyError(
            f"Grid de parâmetros não definido para '{algorithm_name}'. "
            f"Disponíveis: {list(PARAM_GRIDS.keys())}"
        )

    param_grid = PARAM_GRIDS[algorithm_name]
    model_factory = get_model_factory(algorithm_name)

    # Gerar todas as combinações
    param_list = [
        dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())
    ]

    total = len(param_list)
    logger.info(
        "Grid Search para '%s' — %d combinações de hiperparâmetros.",
        algorithm_name,
        total,
    )

    # Execução paralela com progress bar
    with _tqdm_joblib(tqdm(desc="Treinando modelos", total=total)):
        results: list[EvaluationResult] = Parallel(n_jobs=-1)(
            delayed(evaluate_model)(
                model_factory,
                params,
                X_train,
                X_val,
                y_val_transformed,
                verbose=False,
            )
            for params in param_list
        )

    best = max(results, key=lambda r: r.f1)
    logger.info("Melhor F1 encontrado: %.4f | Params: %s", best.f1, best.params)
    return best
