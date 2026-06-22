from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.utils import resample, shuffle

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
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
    stat_filter: Optional["StatisticalFilter"] = None
    pca: Optional[Any] = None


@dataclass
class StatisticalFilter:
    kept_columns: pd.Index
    dropped_low_variance: list[str]
    dropped_duplicates: list[str]
    dropped_high_correlation: list[str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        missing = [c for c in self.kept_columns if c not in X.columns]
        if missing:
            logger.warning(
                "StatisticalFilter.transform: %d coluna(s) do treino ausente(s) "
                "na entrada (serão preenchidas pelo imputer): %s",
                len(missing),
                missing[:5] + (["..."] if len(missing) > 5 else []),
            )
        return X.reindex(columns=self.kept_columns)


def load_arff(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    logger.info("Carregando dataset: %s", path)
    data, _meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    original_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")
    removed = original_rows - len(df)
    if removed > 0:
        logger.info("Removidas %d linhas com valores inválidos.", removed)
    return df, removed


def clean_features(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X.replace([np.inf, -np.inf], np.nan)


def apply_imputation(
    X: np.ndarray | pd.DataFrame,
    imputer: SimpleImputer,
    *,
    fit: bool = False,
) -> np.ndarray:
    if fit:
        result = imputer.fit_transform(X)
    else:
        result = imputer.transform(X)

    return np.where(result < 0, 0, result)


def apply_statistical_filters(
    X: pd.DataFrame,
    *,
    variance_threshold: float = 0.0,
    correlation_threshold: float = 0.95,
) -> StatisticalFilter:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    initial_cols = X.shape[1]
    logger.info("Filtros estatísticos — entrada: %d colunas.", initial_cols)

    variances = X.var(axis=0, numeric_only=True)
    low_var_cols = variances[variances <= variance_threshold].index.tolist()
    X_step1 = X.drop(columns=low_var_cols)
    logger.info(
        "  [1] Variância <= %g: %d colunas descartadas%s",
        variance_threshold,
        len(low_var_cols),
        (
            f" ({low_var_cols[:5]}{'...' if len(low_var_cols) > 5 else ''})"
            if low_var_cols
            else ""
        ),
    )

    duplicated_mask = X_step1.T.duplicated(keep="first")
    duplicate_cols = X_step1.columns[duplicated_mask].tolist()
    X_step2 = X_step1.drop(columns=duplicate_cols)
    logger.info(
        "  [2] Duplicatas literais: %d colunas descartadas%s",
        len(duplicate_cols),
        f" ({duplicate_cols})" if duplicate_cols else "",
    )

    corr_matrix = X_step2.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [
        col for col in upper.columns if (upper[col] > correlation_threshold).any()
    ]
    X_step3 = X_step2.drop(columns=high_corr_cols)
    logger.info(
        "  [3] |ρ| > %.2f: %d colunas descartadas%s",
        correlation_threshold,
        len(high_corr_cols),
        (
            f" ({high_corr_cols[:5]}{'...' if len(high_corr_cols) > 5 else ''})"
            if high_corr_cols
            else ""
        ),
    )

    kept = X_step3.columns
    logger.info(
        "Filtros estatísticos — saída: %d colunas (%d removidas, %.1f%% do total).",
        len(kept),
        initial_cols - len(kept),
        100 * (initial_cols - len(kept)) / initial_cols if initial_cols else 0.0,
    )

    return StatisticalFilter(
        kept_columns=kept,
        dropped_low_variance=low_var_cols,
        dropped_duplicates=duplicate_cols,
        dropped_high_correlation=high_corr_cols,
    )


def apply_contamination(
    X: pd.DataFrame,
    y: pd.Series,
    target_contamination: Optional[float],
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
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


def transform_labels(y: np.ndarray) -> np.ndarray:
    return np.where(y == 0, 1, -1)


def prepare_splits(
    df: pd.DataFrame,
    label_column: str = "Label",
    *,
    novelty: bool = True,
    algorithm: str = "isolation_forest",
    contamination: Optional[float] = None,
    train_size: float = 0.6,
    random_state: int = 42,
    apply_filters: bool = True,
    variance_threshold: float = 0.0,
    correlation_threshold: float = 0.95,
    pca_reduce: Optional[int] = None,
    domain_features: Optional[list[str]] = None,
) -> SplitData:
    from niod.utils.domain_features import add_domain_features

    if domain_features:
        df = add_domain_features(df, features=domain_features)
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[label_column])
    y = df[label_column]
    feature_columns = X.columns

    test_ratio_in_temp = 0.75

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

    y_val_transformed = transform_labels(y_val)
    y_test_transformed = transform_labels(y_test)

    X_train = clean_features(X_train)
    X_val = clean_features(X_val)
    X_test = clean_features(X_test)

    if list(X_train.columns) != list(feature_columns):
        X_train.columns = feature_columns
        X_val.columns = feature_columns
        X_test.columns = feature_columns

    stat_filter: Optional[StatisticalFilter] = None
    if apply_filters:
        stat_filter = apply_statistical_filters(
            X_train.fillna(0),
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
        )
        X_train = stat_filter.transform(X_train)
        X_val = stat_filter.transform(X_val)
        X_test = stat_filter.transform(X_test)
        feature_columns = stat_filter.kept_columns

    imputer = SimpleImputer(strategy="mean")
    X_train = apply_imputation(X_train, imputer, fit=True)
    X_val = apply_imputation(X_val, imputer, fit=False)
    X_test = apply_imputation(X_test, imputer, fit=False)

    pca_model = None
    if pca_reduce is not None and pca_reduce > 0:
        from sklearn.decomposition import PCA

        n_features_in = X_train.shape[1]
        if pca_reduce >= n_features_in:
            logger.warning(
                "pca_reduce=%d >= n_features=%d. Pulando PCA.",
                pca_reduce,
                n_features_in,
            )
        else:
            pca_model = PCA(n_components=pca_reduce, random_state=random_state)
            X_train = pca_model.fit_transform(X_train)
            X_val = pca_model.transform(X_val)
            X_test = pca_model.transform(X_test)
            explained = pca_model.explained_variance_ratio_.sum()
            logger.info(
                "PCA aplicado: %d → %d componentes (%.2f%% da variância retida).",
                n_features_in,
                pca_reduce,
                explained * 100,
            )
            feature_columns = pd.Index([f"PC{i+1}" for i in range(pca_reduce)])

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
        stat_filter=stat_filter,
        pca=pca_model,
    )


def _split_novelty(
    df: pd.DataFrame,
    label_column: str,
    train_size: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, np.ndarray]:
    from sklearn.model_selection import train_test_split

    temp_size = 1.0 - train_size

    X_normais = df[df[label_column] == 0].drop(columns=[label_column]).values
    X_outliers = df[df[label_column] == 1].drop(columns=[label_column]).values
    y_normais = df[df[label_column] == 0][label_column].values
    y_outliers = df[df[label_column] == 1][label_column].values

    logger.info("Normais: %d | Outliers: %d", len(X_normais), len(X_outliers))

    X_train, X_temp_n, y_train, y_temp_n = train_test_split(
        X_normais, y_normais, test_size=temp_size, random_state=random_state
    )
    X_val_n, X_test_n, y_val_n, y_test_n = train_test_split(
        X_temp_n, y_temp_n, test_size=test_ratio, random_state=random_state
    )

    _, X_temp_o, _, y_temp_o = train_test_split(
        X_outliers, y_outliers, test_size=temp_size, random_state=random_state
    )
    X_val_o, X_test_o, y_val_o, y_test_o = train_test_split(
        X_temp_o, y_temp_o, test_size=test_ratio, random_state=random_state
    )

    X_val = np.concatenate((X_val_n, X_val_o))
    y_val = np.concatenate((y_val_n, y_val_o))
    X_test = np.concatenate((X_test_n, X_test_o))
    y_test = np.concatenate((y_test_n, y_test_o))

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


def load_extra_normal(
    extra_path: Path,
    stat_filter: Optional["StatisticalFilter"],
    imputer: SimpleImputer,
    feature_columns: pd.Index,
    *,
    domain_features: Optional[list[str]] = None,
    pca: Optional[Any] = None,
    label_column: str = "Label",
) -> np.ndarray:
    from niod.utils.domain_features import add_domain_features

    logger.info("Carregando normais extras de: %s", extra_path)
    df = load_arff(extra_path)

    if domain_features:
        df = add_domain_features(df, features=domain_features)

    if label_column in df.columns:
        df = df[df[label_column] == 0].drop(columns=[label_column])
    else:
        logger.warning("Coluna '%s' não encontrada em %s — usando todas as linhas.", label_column, extra_path)

    if len(df) == 0:
        logger.warning("Nenhuma amostra normal encontrada em %s.", extra_path)
        return np.empty((0, len(feature_columns)))

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")

    available = [c for c in feature_columns if c in df.columns]
    df = df[available]

    X = clean_features(df)

    if stat_filter is not None:
        X = stat_filter.transform(X)
    X = apply_imputation(X, imputer, fit=False)
    if pca is not None:
        X = pca.transform(X)

    X_arr = np.asarray(X)
    logger.info("Normais extras carregadas: %d amostras", len(X_arr))
    return X_arr


def prepare_data_for_visualization(
    file_path: Path,
    columns_ref: pd.Index,
    fill_values: pd.Series,
    label_desc: str,
    sample_size: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_arff(file_path)
    target_col = df.columns[-1]

    if df[target_col].dtype == object:
        df[target_col] = df[target_col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
        )

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
