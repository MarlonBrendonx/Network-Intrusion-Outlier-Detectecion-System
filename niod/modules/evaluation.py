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


@dataclass
class EvaluationResult:
    f1: float
    params: dict[str, Any]
    report: str
    confusion_matrix: np.ndarray
    pipeline: Pipeline


def evaluate_model(
    model_factory: ModelFactory,
    params: dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test_transformed: np.ndarray,
    *,
    verbose: bool = True,
) -> EvaluationResult:
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


def hyperparameters_search(
    algorithm_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val_transformed: np.ndarray,
    *,
    n_jobs: int = -1,
) -> EvaluationResult:
    if algorithm_name not in PARAM_GRIDS:
        raise KeyError(
            f"Grid de parâmetros não definido para '{algorithm_name}'. "
            f"Disponíveis: {list(PARAM_GRIDS.keys())}"
        )

    param_grid = PARAM_GRIDS[algorithm_name]
    model_factory = get_model_factory(algorithm_name)

    param_list = [
        dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())
    ]

    total = len(param_list)
    logger.info(
        "Grid Search para '%s' — %d combinações de hiperparâmetros.",
        algorithm_name,
        total,
    )

    with _tqdm_joblib(tqdm(desc="Treinando modelos", total=total)):
        results: list[EvaluationResult] = Parallel(n_jobs=n_jobs)(
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
