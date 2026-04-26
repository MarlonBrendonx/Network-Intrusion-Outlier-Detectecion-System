"""
Funções de carregamento, limpeza e pré-processamento de dados.

Princípios:
- Imputer é fitado SOMENTE no conjunto de treino.
- Nenhuma informação do teste/validação vaza para o treino.
- Todas as transformações são reproduzíveis via random_state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.utils import resample, shuffle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Containers de dados tipados
# ---------------------------------------------------------------------------
@dataclass
class SplitData:
    """Contém os conjuntos de treino, validação e teste já divididos."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_val_transformed: np.ndarray
    y_test_transformed: np.ndarray
    feature_columns: pd.Index
    imputer: SimpleImputer


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
def load_arff(path: Path) -> pd.DataFrame:
    """Carrega um arquivo .arff e retorna um DataFrame limpo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    logger.info("Carregando dataset: %s", path)
    data, _meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove infinitos e NaN do DataFrame bruto.

    Returns:
        Tupla (df_limpo, quantidade_linhas_removidas).
    """
    original_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")
    removed = original_rows - len(df)
    if removed > 0:
        logger.info("Removidas %d linhas com valores inválidos.", removed)
    return df, removed


def clean_features(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    """Converte tipos e substitui infinitos por NaN (pré-imputação)."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X.replace([np.inf, -np.inf], np.nan)


def apply_imputation(
    X: np.ndarray | pd.DataFrame,
    imputer: SimpleImputer,
    *,
    fit: bool = False,
) -> np.ndarray:
    """
    Aplica imputação de forma controlada.

    Args:
        X: Dados a imputar.
        imputer: Instância de SimpleImputer.
        fit: Se True, faz fit_transform (SOMENTE para treino).

    Returns:
        Array numpy imputado com negativos zerados.
    """
    if fit:
        result = imputer.fit_transform(X)
    else:
        result = imputer.transform(X)

    # Zerar valores negativos (coerência com dados de rede)
    return np.where(result < 0, 0, result)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de interação para detecção de anomalias de rede.

    Retorna o DataFrame com colunas adicionais, sem modificar o original.
    """
    df = df.copy()
    epsilon = 1e-6

    required_cols = {
        "Fwd Packet Length Min",
        "Fwd Packet Length Max",
        "Fwd Header Length",
        "Subflow Fwd Bytes",
        "Subflow Fwd Packets",
        "Flow Duration",
        "Flow IAT Mean",
        "Flow IAT Max",
    }
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning(
            "Colunas ausentes para feature engineering: %s. Pulando.",
            missing,
        )
        return df

    df["Eng_Packet_Shape"] = df["Fwd Packet Length Min"] / (
        df["Fwd Packet Length Max"] + epsilon
    )
    df["Eng_Fwd_Header_Load"] = df["Fwd Header Length"] / (
        df["Subflow Fwd Bytes"] + epsilon
    )
    df["Eng_Fwd_Velocity"] = df["Subflow Fwd Packets"] / (df["Flow Duration"] + epsilon)
    df["Eng_IAT_Regularity"] = df["Flow IAT Mean"] / (df["Flow IAT Max"] + epsilon)

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    return df


# ---------------------------------------------------------------------------
# Aplicação de contaminação controlada
# ---------------------------------------------------------------------------
def apply_contamination(
    X: pd.DataFrame,
    y: pd.Series,
    target_contamination: Optional[float],
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Ajusta a proporção de outliers para atingir a contaminação desejada.

    Faz downsample da classe minoritária (outlier) para atingir a proporção
    exata, preservando todos os dados benignos.
    """
    if target_contamination is None:
        return X, y

    X_benign = X[y == 0]
    X_outlier = X[y == 1]
    y_benign = y[y == 0]

    n_benign = len(X_benign)
    n_outliers_needed = int(
        (target_contamination * n_benign) / (1 - target_contamination)
    )

    if n_outliers_needed > len(X_outlier):
        logger.warning(
            "Dados insuficientes para contaminação %.2f. "
            "Usando máximo disponível (%d de %d necessários).",
            target_contamination,
            len(X_outlier),
            n_outliers_needed,
        )
        n_outliers_needed = len(X_outlier)

    X_outlier_sampled = resample(
        X_outlier,
        replace=False,
        n_samples=n_outliers_needed,
        random_state=random_state,
    )
    y_outlier_sampled = pd.Series(
        [1] * n_outliers_needed, index=X_outlier_sampled.index
    )

    X_final = pd.concat([X_benign, X_outlier_sampled])
    y_final = pd.concat([y_benign, y_outlier_sampled])

    return resample(X_final, y_final, replace=False, random_state=random_state)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def transform_labels(y: np.ndarray) -> np.ndarray:
    """Converte labels 0/1 → 1/-1 (formato sklearn anomaly detection)."""
    return np.where(y == 0, 1, -1)


# ---------------------------------------------------------------------------
# Split principal
# ---------------------------------------------------------------------------
def prepare_splits(
    df: pd.DataFrame,
    label_column: str = "Label",
    *,
    novelty: bool = True,
    algorithm: str = "isolation_forest",
    contamination: Optional[float] = None,
    train_size: float = 0.6,
    random_state: int = 42,
) -> SplitData:
    """
    Divide os dados em treino/validação/teste com tratamento correto
    para novelty detection e detecção de outliers.

    Garante:
    - Sem data leakage: imputer fitado apenas no treino.
    - Stratificação quando aplicável.
    - Shuffle dos conjuntos mistos.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[label_column])
    y = df[label_column]
    feature_columns = X.columns

    test_ratio_in_temp = 0.75  # 75% de (1 - train_size) → ~30% do total

    if novelty or algorithm == "svm":
        splits = _split_novelty(
            df, label_column, train_size, test_ratio_in_temp, random_state
        )
    else:
        splits = _split_standard(
            X, y, train_size, test_ratio_in_temp, random_state, contamination
        )

    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    # Log de proporções
    logger.info(
        "X_train: %s", X_train.shape if hasattr(X_train, "shape") else len(X_train)
    )
    logger.info("X_val:   %s", X_val.shape if hasattr(X_val, "shape") else len(X_val))
    logger.info(
        "X_test:  %s", X_test.shape if hasattr(X_test, "shape") else len(X_test)
    )
    logger.info(
        "Proporção outliers — Treino: %.4f | Val: %.4f | Teste: %.4f",
        np.mean(y_train),
        np.mean(y_val),
        np.mean(y_test),
    )

    # Transformar labels para formato sklearn
    y_val_transformed = transform_labels(y_val)
    y_test_transformed = transform_labels(y_test)

    # Limpeza e imputação (sem leakage)
    X_train = clean_features(X_train)
    X_val = clean_features(X_val)
    X_test = clean_features(X_test)

    imputer = SimpleImputer(strategy="mean")
    X_train = apply_imputation(X_train, imputer, fit=True)
    X_val = apply_imputation(X_val, imputer, fit=False)
    X_test = apply_imputation(X_test, imputer, fit=False)

    return SplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_val_transformed=y_val_transformed,
        y_test_transformed=y_test_transformed,
        feature_columns=feature_columns,
        imputer=imputer,
    )


def _split_novelty(
    df: pd.DataFrame,
    label_column: str,
    train_size: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, np.ndarray]:
    """Split para novelty detection: treino contém apenas amostras normais."""
    from sklearn.model_selection import train_test_split

    temp_size = 1.0 - train_size

    X_normais = df[df[label_column] == 0].drop(columns=[label_column]).values
    X_outliers = df[df[label_column] == 1].drop(columns=[label_column]).values
    y_normais = df[df[label_column] == 0][label_column].values
    y_outliers = df[df[label_column] == 1][label_column].values

    logger.info("Normais: %d | Outliers: %d", len(X_normais), len(X_outliers))

    # Normais: 60% treino, 10% val, 30% teste
    X_train, X_temp_n, y_train, y_temp_n = train_test_split(
        X_normais, y_normais, test_size=temp_size, random_state=random_state
    )
    X_val_n, X_test_n, y_val_n, y_test_n = train_test_split(
        X_temp_n, y_temp_n, test_size=test_ratio, random_state=random_state
    )

    # Outliers: descarte 60%, 10% val, 30% teste
    _, X_temp_o, _, y_temp_o = train_test_split(
        X_outliers, y_outliers, test_size=temp_size, random_state=random_state
    )
    X_val_o, X_test_o, y_val_o, y_test_o = train_test_split(
        X_temp_o, y_temp_o, test_size=test_ratio, random_state=random_state
    )

    # Combinar val e teste (normais + outliers)
    X_val = np.concatenate((X_val_n, X_val_o))
    y_val = np.concatenate((y_val_n, y_val_o))
    X_test = np.concatenate((X_test_n, X_test_o))
    y_test = np.concatenate((y_test_n, y_test_o))

    # Shuffle dos conjuntos mistos
    X_val, y_val = shuffle(X_val, y_val, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def _split_standard(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
    test_ratio: float,
    random_state: int,
    contamination: Optional[float],
) -> dict[str, np.ndarray | pd.DataFrame]:
    """Split padrão com estratificação e contaminação controlada."""
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, random_state=random_state, stratify=y_temp
    )

    if contamination is not None:
        X_train, y_train = apply_contamination(
            X_train, y_train, contamination, random_state
        )
        X_val, y_val = apply_contamination(X_val, y_val, contamination, random_state)
        X_test, y_test = apply_contamination(
            X_test, y_test, contamination, random_state
        )

    total = len(X_train) + len(X_val) + len(X_test)
    logger.info(
        "Split — Treino: %d (%.0f%%) | Val: %d (%.0f%%) | Teste: %d (%.0f%%)",
        len(X_train),
        100 * len(X_train) / total,
        len(X_val),
        100 * len(X_val) / total,
        len(X_test),
        100 * len(X_test) / total,
    )

    return {
        "X_train": X_train.values if hasattr(X_train, "values") else X_train,
        "X_val": X_val.values if hasattr(X_val, "values") else X_val,
        "X_test": X_test.values if hasattr(X_test, "values") else X_test,
        "y_train": y_train.values if hasattr(y_train, "values") else y_train,
        "y_val": y_val.values if hasattr(y_val, "values") else y_val,
        "y_test": y_test.values if hasattr(y_test, "values") else y_test,
    }


# ---------------------------------------------------------------------------
# Preparação para visualização (PCA/UMAP)
# ---------------------------------------------------------------------------
def prepare_data_for_visualization(
    file_path: Path,
    columns_ref: pd.Index,
    fill_values: pd.Series,
    label_desc: str,
    sample_size: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega e prepara dados de um dataset externo para visualização.

    Alinha colunas com o dataset de referência e aplica amostragem.

    Returns:
        Tupla (X_normais, X_ataques).
    """
    df = load_arff(file_path)
    target_col = df.columns[-1]

    # Decodificar labels se necessário
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
        )

    # Identificar normais vs ataques
    if pd.api.types.is_numeric_dtype(df[target_col]):
        mask_normal = df[target_col] == 0
    else:
        mask_normal = df[target_col] == "BENIGN"

    X_local = df.drop(columns=[target_col])
    X_local = X_local.reindex(columns=columns_ref)
    X_local = X_local.replace([np.inf, -np.inf], np.nan)
    X_local = X_local.fillna(fill_values)
    X_local = X_local.clip(lower=0)

    X_norm = X_local[mask_normal]
    X_att = X_local[~mask_normal]

    # Amostragem para performance
    if len(X_norm) > sample_size:
        X_norm = X_norm.sample(n=sample_size, random_state=42)
    if len(X_att) > sample_size:
        X_att = X_att.sample(n=sample_size, random_state=42)
    elif len(X_att) == 0:
        logger.warning("Nenhum ataque encontrado em %s!", label_desc)

    logger.info(
        "%s — Normais: %d | Ataques: %d",
        label_desc,
        len(X_norm),
        len(X_att),
    )
    return X_norm, X_att
