"""
Teste de generalização (Concept Drift).

Avalia o modelo treinado em um dataset diferente do usado no treino,
medindo a capacidade de generalização contra novos padrões de ataque.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from niod.utils.data import (
    clean_features,
    apply_imputation,
    create_interaction_features,
    load_arff,
    transform_labels,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneralizationResult:
    """Resultado do teste de generalização."""

    report: str
    confusion_matrix: np.ndarray
    recall_attack: float
    drift_detected: bool
    n_samples: int


def test_generalization(
    pipeline: Pipeline,
    generalization_path: Path,
    imputer: SimpleImputer,
    *,
    feature_columns: list[str] | pd.Index | None = None,
    drift_threshold: float = 0.1,
) -> GeneralizationResult:
    """
    Testa o modelo treinado contra um dataset de generalização.

    Args:
        pipeline: Pipeline treinado (scaler + modelo).
        generalization_path: Caminho do dataset de generalização.
        imputer: Imputer fitado no treino (para manter consistência).
        feature_columns: Colunas a selecionar (se aplicável).
        drift_threshold: Limiar de recall abaixo do qual se detecta drift.

    Returns:
        GeneralizationResult com métricas e diagnóstico de drift.
    """
    logger.info("Carregando dataset de generalização: %s", generalization_path)
    df = load_arff(generalization_path)

    # Aplicar feature engineering se as colunas existirem
    # df = create_interaction_features(df)

    # Selecionar colunas específicas se fornecidas
    if feature_columns is not None:
        available = [c for c in feature_columns if c in df.columns]
        missing = set(feature_columns) - set(available)
        if missing:
            logger.warning("Colunas ausentes no dataset de generalização: %s", missing)
        df = df[available + ["Label"]]

    # Limpeza
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")

    X = df.drop(columns=["Label"])
    y_raw = df["Label"]
    y_true = transform_labels(y_raw.values)

    # Pré-processamento consistente com treino
    X_clean = clean_features(X)
    X_clean = apply_imputation(X_clean, imputer, fit=False)

    logger.info("Realizando predição em %d amostras...", len(X_clean))
    y_pred = pipeline.predict(X_clean)

    # Métricas
    report = classification_report(
        y_true,
        y_pred,
        labels=[1, -1],
        target_names=["Normal (1)", "Ataque (-1)"],
        digits=4,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])

    # Recall de ataque (evitar divisão por zero)
    attack_tp = cm[1][1]
    attack_fn = cm[1][0]
    recall_attack = (
        attack_tp / (attack_tp + attack_fn) if (attack_tp + attack_fn) > 0 else 0.0
    )

    drift_detected = recall_attack < drift_threshold

    if drift_detected:
        logger.warning(
            "CONCEPT DRIFT DETECTADO — Recall de ataque: %.2f%% (< %.0f%%)",
            recall_attack * 100,
            drift_threshold * 100,
        )
    else:
        logger.info(
            "Generalização funcional — Recall de ataque: %.2f%%",
            recall_attack * 100,
        )

    return GeneralizationResult(
        report=report,
        confusion_matrix=cm,
        recall_attack=recall_attack,
        drift_detected=drift_detected,
        n_samples=len(X_clean),
    )
