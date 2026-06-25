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

# Algorithms whose `contamination` only sets the threshold (offset_), not the model:
# we sweep the candidates via score_samples from a SINGLE fit, instead of
# refitting the entire model per contamination value.
_CONTAMINATION_SWEEP_ALGOS = {"isolation_forest", "lof"}

# OneClassSVM (libsvm RBF) is O(n²)–O(n³): infeasible on the ~250k normal training samples.
# We subsample to a tractable size (logged) so the algorithm is
# comparable to the others in the benchmark. Raise this cap if you have time/memory.
_SVM_MAX_FIT_SAMPLES = 20000


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
        logger.debug("[LOF] Outlier Detection mode (fit_predict on test)")
        y_pred = pipeline.fit_predict(X_test)
    else:
        is_svm = estimator.__class__.__name__ == "OneClassSVM"
        if is_svm and len(X_train) > _SVM_MAX_FIT_SAMPLES:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), _SVM_MAX_FIT_SAMPLES, replace=False)
            logger.info(
                "[OneClassSVM] Subsampling training set: %d → %d samples (tractability).",
                len(X_train),
                _SVM_MAX_FIT_SAMPLES,
            )
            X_train = X_train[idx]
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


def _evaluate_with_contamination_sweep(
    model_factory: ModelFactory,
    params: dict[str, Any],
    contamination_candidates: list[float],
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val_transformed: np.ndarray,
) -> list[EvaluationResult]:
    """Fit ONCE and evaluate several `contamination` values via score_samples.

    For IsolationForest/LOF-novelty, `contamination` only shifts the threshold
    (`offset_ = percentile of the training scores`); the trees / the neighbor
    graph are independent of it. Therefore a single fit serves all candidates:
    `predict == -1` ⟺ `score_samples(X) < offset_`. This avoids refitting the
    entire model per contamination value.
    """
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("estimator", model_factory(**params)),
        ]
    )
    pipeline.fit(X_train)

    estimator = pipeline.named_steps["estimator"]
    # Percentile base: LOF stores the training scores in
    # negative_outlier_factor_ (what sklearn uses for the offset_); IF does not
    # store them, so we recompute them via score_samples on the training set.
    if hasattr(estimator, "negative_outlier_factor_"):
        train_scores = estimator.negative_outlier_factor_
    else:
        train_scores = pipeline.score_samples(X_train)

    val_scores = pipeline.score_samples(X_val)

    results: list[EvaluationResult] = []
    for contamination in contamination_candidates:
        offset = np.percentile(train_scores, 100.0 * contamination)
        y_pred = np.where(val_scores < offset, -1, 1)

        f1 = f1_score(y_val_transformed, y_pred, labels=[1, -1], average="macro")
        full_params = {**params, "contamination": contamination}
        report = classification_report(
            y_val_transformed,
            y_pred,
            labels=[1, -1],
            target_names=["Normal (1)", "Outlier (-1)"],
            output_dict=False,
        )
        cm = confusion_matrix(y_val_transformed, y_pred, labels=[1, -1])
        results.append(
            EvaluationResult(
                f1=f1,
                params=full_params,
                report=report,
                confusion_matrix=cm,
                pipeline=pipeline,
            )
        )
    return results


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
            f"Parameter grid not defined for '{algorithm_name}'. "
            f"Available: {list(PARAM_GRIDS.keys())}"
        )

    param_grid = dict(PARAM_GRIDS[algorithm_name])
    model_factory = get_model_factory(algorithm_name)

    # `contamination` is threshold-only for IF/LOF: we remove it from the cartesian
    # product and sweep it via score_samples from a single fit per combination.
    sweep_contamination = (
        algorithm_name in _CONTAMINATION_SWEEP_ALGOS and "contamination" in param_grid
    )
    contamination_candidates: list[float] = []
    if sweep_contamination:
        contamination_candidates = list(param_grid.pop("contamination"))

    param_list = [
        dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())
    ]

    total_fits = len(param_list)
    total_configs = total_fits * (len(contamination_candidates) or 1)
    logger.info(
        "Grid Search for '%s' — %d configurations (%d fits%s).",
        algorithm_name,
        total_configs,
        total_fits,
        (
            f"; contamination {contamination_candidates} swept without refit"
            if sweep_contamination
            else ""
        ),
    )

    with _tqdm_joblib(tqdm(desc="Training models", total=total_fits)):
        if sweep_contamination:
            nested: list[list[EvaluationResult]] = Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_with_contamination_sweep)(
                    model_factory,
                    params,
                    contamination_candidates,
                    X_train,
                    X_val,
                    y_val_transformed,
                )
                for params in param_list
            )
            results: list[EvaluationResult] = [r for sub in nested for r in sub]
        else:
            results = Parallel(n_jobs=n_jobs)(
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
    logger.info("Best F1 found: %.4f | Params: %s", best.f1, best.params)
    return best
