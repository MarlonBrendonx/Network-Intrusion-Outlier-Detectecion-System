from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Any

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from xgboost import XGBClassifier

from niod.config.settings import CLASSIFICATION_PARAM_GRIDS

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    f1: float
    params: dict[str, Any]
    report: str
    confusion_matrix: np.ndarray
    model: Any


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
        raise KeyError(f"Classifier '{name}' not found. Available: {available}")
    return CLASSIFICATION_MODEL_REGISTRY[name]


def build_classifier(
    model_factory,
    params: dict[str, Any],
    *,
    use_smote: bool = True,
    random_state: int = 42,
):
    """Build the estimator to be trained.

    When ``use_smote`` is True, the classifier is wrapped in an
    ``imblearn.pipeline.Pipeline`` (SMOTE → estimator). SMOTE resampling
    runs ONLY during ``.fit()``; ``.predict()`` ignores it entirely. Thus
    the balancing never reaches val/test — the leakage is impossible by
    construction, not just by usage convention.
    """
    estimator = model_factory(**params)
    if not use_smote:
        return estimator
    return ImbPipeline(
        [
            ("smote", SMOTE(random_state=random_state)),
            ("estimator", estimator),
        ]
    )


def evaluate_classifier(
    model_factory,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    verbose: bool = True,
    use_smote: bool = True,
    random_state: int = 42,
) -> ClassificationResult:
    model = build_classifier(
        model_factory, params, use_smote=use_smote, random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, labels=[0, 1], average="macro")

    if verbose:
        logger.info("PARAMS: %s  →  F1: %.4f", params, f1)

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal (0)", "Attack (1)"],
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


def hyperparameters_search_classifier(
    algorithm_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_jobs: int = 8,
    use_smote: bool = True,
    random_state: int = 42,
) -> ClassificationResult:
    if algorithm_name not in CLASSIFICATION_PARAM_GRIDS:
        raise KeyError(
            f"Grid not defined for '{algorithm_name}'. "
            f"Available: {list(CLASSIFICATION_PARAM_GRIDS.keys())}"
        )

    param_grid = CLASSIFICATION_PARAM_GRIDS[algorithm_name]
    model_factory = get_classifier_factory(algorithm_name)

    param_list = [
        dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())
    ]

    total = len(param_list)
    logger.info(
        "Grid Search for '%s' — %d hyperparameter combinations.",
        algorithm_name,
        total,
    )

    with _tqdm_joblib(tqdm(desc="Training classifiers", total=total)):
        results: list[ClassificationResult] = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_classifier)(
                model_factory,
                params,
                X_train,
                y_train,
                X_val,
                y_val,
                verbose=False,
                use_smote=use_smote,
                random_state=random_state,
            )
            for params in param_list
        )

    best = max(results, key=lambda r: r.f1)
    logger.info("Best F1 found: %.4f | Params: %s", best.f1, best.params)
    return best
