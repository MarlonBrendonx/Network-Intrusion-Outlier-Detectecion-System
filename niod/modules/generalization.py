"""
Teste de generalização (Concept Drift).

Avalia o modelo treinado em um dataset diferente do usado no treino,
medindo a capacidade de generalização contra novos padrões de ataque.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from niod.utils.data import (
    StatisticalFilter,
    clean_features,
    apply_imputation,
    load_arff,
    transform_labels,
)

logger = logging.getLogger(__name__)


@dataclass
class KSDriftResult:
    """Resultado da análise KS por feature entre treino e generalização."""

    statistics: dict[str, float]
    drifted_features: list[str]
    drift_ratio: float


@dataclass
class GeneralizationResult:
    """Resultado do teste de generalização."""

    report: str
    confusion_matrix: np.ndarray
    recall_attack: float
    drift_detected: bool
    n_samples: int
    ks_result: KSDriftResult | None = field(default=None)


def ks_drift_analysis(
    X_train: np.ndarray,
    X_gen: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.3,
) -> KSDriftResult:
    """
    Aplica o teste KS feature-a-feature entre tráfego normal do treino e da generalização.

    Args:
        X_train: Amostras normais do treino (já pré-processadas).
        X_gen: Amostras normais do dataset de generalização (mesmo espaço).
        feature_names: Nomes das features (ou "PC_i" quando PCA foi aplicado).
        threshold: KS acima deste valor sinaliza drift naquela feature.

    Returns:
        KSDriftResult com estatísticas por feature e resumo de drift.
    """
    statistics: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        stat, _ = ks_2samp(X_train[:, i], X_gen[:, i])
        statistics[name] = round(float(stat), 4)

    drifted = [name for name, stat in statistics.items() if stat > threshold]
    drift_ratio = len(drifted) / len(feature_names) if feature_names else 0.0

    return KSDriftResult(
        statistics=statistics,
        drifted_features=drifted,
        drift_ratio=drift_ratio,
    )


def test_generalization(
    pipeline: Pipeline,
    generalization_path: Path,
    imputer: SimpleImputer,
    *,
    feature_columns: list[str] | pd.Index | None = None,
    stat_filter: StatisticalFilter | None = None,
    pca: object | None = None,
    domain_features: list[str] | None = None,
    drift_threshold: float = 0.1,
    X_train_normal: np.ndarray | None = None,
    ks_feature_names: list[str] | None = None,
    ks_threshold: float = 0.3,
) -> GeneralizationResult:
    """
    Testa o modelo treinado contra um dataset de generalização.

    Args:
        pipeline: Pipeline treinado (scaler + modelo).
        generalization_path: Caminho do dataset de generalização.
        imputer: Imputer fitado no treino (para manter consistência).
        feature_columns: Colunas a selecionar (se aplicável). Ignorado quando
            `stat_filter` é fornecido, pois o filtro já define as colunas.
        stat_filter: Filtro estatístico ajustado no treino. Se fornecido, suas
            colunas são aplicadas ANTES da imputação, garantindo consistência
            com o espaço de features usado para treinar o modelo.
        pca: PCA fitado no treino. Se fornecido, é aplicado APÓS a imputação,
            transformando os dados para o mesmo espaço componente-principal
            usado no treino do modelo.
        domain_features: Lista de features de domínio a adicionar antes do
            filtro. Deve ser idêntica à usada no treino para coerência.
        drift_threshold: Limiar de recall abaixo do qual se detecta drift.
        X_train_normal: Amostras normais do treino (pré-processadas) para análise KS.
            Quando fornecido, compara as distribuições feature-a-feature com as
            amostras normais da generalização via teste KS.
        ks_threshold: KS acima deste valor sinaliza drift naquela feature.

    Returns:
        GeneralizationResult com métricas, diagnóstico de drift e análise KS opcional.
    """
    from niod.utils.domain_features import add_domain_features

    logger.info("Carregando dataset de generalização: %s", generalization_path)
    df = load_arff(generalization_path)

    # Aplica as mesmas features de domínio do treino (puramente determinístico
    # — apenas razões entre colunas existentes, sem learning).
    if domain_features:
        df = add_domain_features(df, features=domain_features)

    # Selecionar colunas: prioridade para stat_filter; fallback em feature_columns
    if stat_filter is not None:
        cols_to_keep = list(stat_filter.kept_columns)
        available = [c for c in cols_to_keep if c in df.columns]
        missing = set(cols_to_keep) - set(available)
        if missing:
            logger.warning(
                "Colunas ausentes no dataset de generalização (stat_filter): %s",
                missing,
            )
        df = df[available + ["Label"]]
    elif feature_columns is not None:
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

    # Pré-processamento consistente com treino.
    # Se stat_filter foi aplicado, reordena colunas no mesmo layout do treino
    # (crucial: sklearn trabalha por POSIÇÃO, não por nome, após fit).
    X_clean = clean_features(X)
    if stat_filter is not None:
        X_clean = stat_filter.transform(X_clean)
    X_clean = apply_imputation(X_clean, imputer, fit=False)
    if pca is not None:
        X_clean = pca.transform(X_clean)

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

    # Análise KS: compara tráfego normal do treino vs generalização por feature
    ks_result: KSDriftResult | None = None
    if X_train_normal is not None and X_train_normal.shape[0] > 0:
        X_gen_normal = X_clean[y_true == 1]
        if len(X_gen_normal) > 0:
            n_features = X_train_normal.shape[1]
            if ks_feature_names and len(ks_feature_names) == n_features:
                feature_names = ks_feature_names
            else:
                feature_names = [f"f_{i}" for i in range(n_features)]

            ks_result = ks_drift_analysis(
                X_train_normal,
                X_gen_normal,
                feature_names,
                threshold=ks_threshold,
            )
            logger.info(
                "KS Drift — Features com drift (KS > %.2f): %d/%d (%.1f%%)",
                ks_threshold,
                len(ks_result.drifted_features),
                len(feature_names),
                ks_result.drift_ratio * 100,
            )

    return GeneralizationResult(
        report=report,
        confusion_matrix=cm,
        recall_attack=recall_attack,
        drift_detected=drift_detected,
        n_samples=len(X_clean),
        ks_result=ks_result,
    )
