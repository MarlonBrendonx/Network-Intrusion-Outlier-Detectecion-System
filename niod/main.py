"""
NIOD — Pipeline principal de detecção de anomalias em tráfego de rede.

Orquestra todo o fluxo: carregamento → split → treino → avaliação →
busca de hiperparâmetros → teste de generalização → visualização UMAP.

Uso:
    python -m niod.main
    python -m niod.main --algorithm svm --no-novelty --no-hyper-search
    python -m niod.main --algorithm lof --skip-umap --all-domain-features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from niod.config.settings import (
    Algorithm,
    DEFAULT_PARAMS,
    ExperimentConfig,
)
from niod.modules.evaluation import (
    EvaluationResult,
    evaluate_model,
    hyperparameters_search,
)
from niod.modules.models import get_model_factory
from niod.utils.data import (
    clean_dataframe,
    load_arff,
    prepare_splits,
)

logger = logging.getLogger("niod")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging(level: str = "INFO") -> None:
    """Configura logging estruturado para todo o projeto."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        # format=log_format,
        # datefmt="%Y-%m-%d %H:%M:%S",
        # handlers=[
        #     logging.StreamHandler(sys.stdout),
        # ],
    )
    # Silenciar logs verbosos de terceiros
    logging.getLogger("joblib").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run_pipeline(config: ExperimentConfig) -> dict:
    """
    Executa o pipeline completo de detecção de anomalias.

    Args:
        config: Configurações do experimento.

    Returns:
        Dicionário com resultados do treino, validação e teste.
    """
    _setup_logging(config.log_level)
    results: dict = {"config": config}

    # ------------------------------------------------------------------
    # 1. Carregamento e limpeza
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("NIOD — Network Intrusion Outlier Detection")
    logger.info("=" * 70)
    logger.info("Algoritmo:   %s", config.algorithm.value)
    logger.info("Novelty:     %s", config.novelty)
    logger.info("Hyper Search: %s", config.hyper_search)
    logger.info("Filtros:     %s", config.apply_filters)
    logger.info("PCA reduce:  %s", config.pca_reduce or "off")
    logger.info("Domain feat: %s", config.domain_features or "off")
    logger.info("Dataset:     %s", config.train_dataset)

    df = load_arff(config.train_dataset)
    df, removed = clean_dataframe(df)
    logger.info("Shape final: %s (removidas %d linhas)", df.shape, removed)

    # ------------------------------------------------------------------
    # 2. Split treino/validação/teste
    # ------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("Dividindo dados...")
    split_data = prepare_splits(
        df,
        label_column="Label",
        novelty=config.novelty,
        algorithm=config.algorithm.value,
        contamination=config.contamination if not config.novelty else None,
        train_size=config.train_size,
        random_state=config.random_state,
        apply_filters=config.apply_filters,
        pca_reduce=config.pca_reduce,
        domain_features=config.domain_features,
    )
    results["split"] = split_data

    # ------------------------------------------------------------------
    # 3. Treino e avaliação
    # ------------------------------------------------------------------
    algorithm_name = config.algorithm.value
    model_factory = get_model_factory(algorithm_name)

    if config.hyper_search:
        # --- Grid Search no conjunto de VALIDAÇÃO ---
        logger.info("-" * 70)
        logger.info("Iniciando busca de hiperparâmetros (%s)...", algorithm_name)

        val_result = hyperparameters_search(
            algorithm_name,
            split_data.X_train,
            split_data.X_val,
            split_data.y_val_transformed,
        )

        _log_evaluation("VALIDAÇÃO", val_result)
        results["validation"] = val_result

        # --- Avaliação final no conjunto de TESTE ---
        logger.info("-" * 70)
        logger.info("Avaliação final no conjunto de TESTE...")

        best_params = val_result.params
        test_result = evaluate_model(
            model_factory,
            best_params,
            split_data.X_train,
            split_data.X_test,
            split_data.y_test_transformed,
        )
    else:
        # --- Sem busca: usar parâmetros default ---
        params = _resolve_default_params(config)
        logger.info("Parâmetros fixos: %s", params)

        test_result = evaluate_model(
            model_factory,
            params,
            split_data.X_train,
            split_data.X_test,
            split_data.y_test_transformed,
        )

    _log_evaluation("TESTE", test_result)
    results["test"] = test_result

    return results


# ---------------------------------------------------------------------------
# Enriquecimento do treino com normais de outro domínio (opcional)
# ---------------------------------------------------------------------------
def enrich_train_with_extra_normal(
    config: ExperimentConfig,
    pipeline_result: dict,
) -> dict:
    """
    Adiciona amostras normais de um dataset extra ao X_train já processado.

    Usa o stat_filter e imputer fittados no treino — sem refitting.
    Só é chamada quando config.extra_normal_dataset está definido.

    Args:
        config: Configurações do experimento.
        pipeline_result: Resultado de run_pipeline().

    Returns:
        pipeline_result com split_data.X_train enriquecido.
    """
    from niod.utils.data import load_extra_normal

    split_data = pipeline_result["split"]
    extra_path = config.extra_normal_dataset

    if extra_path is None or not extra_path.exists():
        logger.warning(
            "Dataset extra não encontrado: %s. Pulando enriquecimento.", extra_path
        )
        return pipeline_result

    logger.info("=" * 70)
    logger.info("ENRIQUECIMENTO DO TREINO COM NORMAIS EXTRAS")
    logger.info("=" * 70)
    logger.info("Fonte: %s", extra_path)

    X_extra = load_extra_normal(
        extra_path=extra_path,
        stat_filter=split_data.stat_filter,
        imputer=split_data.imputer,
        feature_columns=split_data.feature_columns,
        domain_features=config.domain_features,
        pca=split_data.pca,
    )

    if len(X_extra) == 0:
        logger.warning("Nenhuma amostra extra adicionada.")
        return pipeline_result

    original_size = len(split_data.X_train)
    split_data.X_train = np.vstack([split_data.X_train, X_extra])
    # y_train deve crescer junto — todas as extras são normais (label 0)
    split_data.y_train = np.concatenate([split_data.y_train, np.zeros(len(X_extra), dtype=split_data.y_train.dtype)])
    logger.info(
        "X_train: %d → %d amostras (+%d normais de %s)",
        original_size,
        len(split_data.X_train),
        len(X_extra),
        extra_path.stem,
    )

    return pipeline_result


# ---------------------------------------------------------------------------
# Few-shot: adapta o modelo não-supervisionado ao dataset alvo
# ---------------------------------------------------------------------------
def run_few_shot_enrichment(
    config: ExperimentConfig,
    pipeline_result: dict,
) -> dict:
    """
    Adapta o modelo não-supervisionado ao dataset alvo via few-shot.

    Adiciona uma fração das amostras normais do alvo ao treino e retreina o
    pipeline. Os ataques do alvo (mais as normais não usadas) formam o holdout
    armazenado em results['few_shot_holdout'] para a avaliação de generalização,
    garantindo que o modelo não seja avaliado em amostras que viu no treino.

    Nota: LOF em modo não-novelty faz fit_predict no teste e ignora X_train;
    nesse caso o retrein é pulado e apenas o holdout é armazenado.

    Args:
        config: Configurações do experimento (few_shot_dataset, few_shot_ratio).
        pipeline_result: Resultado de run_pipeline().

    Returns:
        pipeline_result atualizado com modelo retreinado e holdout armazenado.
    """
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    from niod.utils.data import apply_imputation, clean_features, load_arff, transform_labels
    from niod.utils.domain_features import add_domain_features

    few_shot_path = config.few_shot_dataset
    ratio = config.few_shot_ratio

    if few_shot_path is None or not few_shot_path.exists():
        logger.warning("Dataset few-shot não encontrado: %s. Pulando.", few_shot_path)
        return pipeline_result

    logger.info("=" * 70)
    logger.info("FEW-SHOT ENRICHMENT — UNSUPERVISED (ratio=%.0f%%)", ratio * 100)
    logger.info("=" * 70)
    logger.info("Fonte: %s", few_shot_path)

    split_data = pipeline_result["split"]
    test_result: EvaluationResult = pipeline_result["test"]

    # Carrega e pré-processa com o mesmo pipeline do treino
    df = load_arff(few_shot_path)
    if config.domain_features:
        df = add_domain_features(df, features=config.domain_features)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    df_normal = df[df["Label"] == 0]
    df_attacks = df[df["Label"] == 1]

    # Separa few-shot (ratio) das normais; ataques ficam 100% no holdout
    n_few_shot = max(1, int(len(df_normal) * ratio))
    df_normal_shuffled = df_normal.sample(frac=1, random_state=config.random_state)
    df_few = df_normal_shuffled.iloc[:n_few_shot]
    df_holdout_normal = df_normal_shuffled.iloc[n_few_shot:]

    df_holdout = pd.concat([df_holdout_normal, df_attacks])

    def _preprocess(df_subset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y = transform_labels(df_subset["Label"].values)
        X = df_subset.drop(columns=["Label"])
        X = clean_features(X)
        if split_data.stat_filter is not None:
            X = split_data.stat_filter.transform(X)
        X = apply_imputation(X, split_data.imputer, fit=False)
        if split_data.pca is not None:
            X = split_data.pca.transform(X)
        return np.asarray(X), y

    X_few, _ = _preprocess(df_few)
    X_holdout, y_holdout = _preprocess(df_holdout)
    X_holdout_normal = X_holdout[y_holdout == 1]

    logger.info(
        "Normais do alvo: %d total → %d few-shot | %d holdout-normal | %d holdout-ataques",
        len(df_normal),
        len(X_few),
        len(df_holdout_normal),
        len(df_attacks),
    )

    # Enriquece X_train com as normais do alvo
    X_train_enriched = np.vstack([split_data.X_train, X_few])
    split_data.X_train = X_train_enriched
    split_data.y_train = np.concatenate(
        [split_data.y_train, np.zeros(len(X_few), dtype=split_data.y_train.dtype)]
    )

    logger.info(
        "X_train: %d → %d amostras (+%d normais de %s)",
        len(X_train_enriched) - len(X_few),
        len(X_train_enriched),
        len(X_few),
        few_shot_path.stem,
    )

    # LOF não-novelty ignora X_train — retrein não faz sentido
    estimator = test_result.pipeline.named_steps["estimator"]
    is_lof = estimator.__class__.__name__ == "LocalOutlierFactor"
    is_novelty = getattr(estimator, "novelty", False)

    if is_lof and not is_novelty:
        logger.warning("LOF modo não-novelty: retrein ignorado (fit_predict no teste).")
    else:
        model_factory = get_model_factory(config.algorithm.value)
        new_pipeline = Pipeline([
            ("scaler", RobustScaler()),
            ("estimator", model_factory(**test_result.params)),
        ])
        new_pipeline.fit(X_train_enriched)

        y_pred = new_pipeline.predict(split_data.X_test)
        f1_new = f1_score(split_data.y_test_transformed, y_pred, labels=[1, -1], average="macro")
        logger.info("F1 no teste original após few-shot: %.4f (antes: %.4f)", f1_new, test_result.f1)

        pipeline_result["test"] = EvaluationResult(
            f1=f1_new,
            params=test_result.params,
            report=classification_report(
                split_data.y_test_transformed, y_pred,
                labels=[1, -1], target_names=["Normal (1)", "Outlier (-1)"],
            ),
            confusion_matrix=confusion_matrix(split_data.y_test_transformed, y_pred, labels=[1, -1]),
            pipeline=new_pipeline,
        )

    pipeline_result["few_shot_holdout"] = {
        "X": X_holdout,
        "y": y_holdout,
        "X_normal": X_holdout_normal,
    }

    return pipeline_result


# ---------------------------------------------------------------------------
# Teste de generalização (opcional)
# ---------------------------------------------------------------------------
def run_generalization(
    config: ExperimentConfig,
    pipeline_result: dict,
) -> dict:
    """
    Executa o teste de generalização em dataset separado.

    Args:
        config: Configurações do experimento.
        pipeline_result: Resultado de run_pipeline().

    Returns:
        Dicionário atualizado com resultado de generalização.
    """
    from niod.modules.generalization import test_generalization, test_generalization_from_arrays

    test_result: EvaluationResult = pipeline_result["test"]
    split_data = pipeline_result["split"]

    logger.info("=" * 70)
    logger.info("TESTE DE GENERALIZAÇÃO (CONCEPT DRIFT)")
    logger.info("=" * 70)

    # Em modo novelty, X_train contém apenas amostras normais — ideal para KS.
    X_train_normal = np.asarray(split_data.X_train)

    # Resolve nomes das features para o relatório KS.
    if split_data.pca is not None:
        n = X_train_normal.shape[1]
        ks_feature_names = [f"PC_{i}" for i in range(n)]
    elif split_data.stat_filter is not None:
        ks_feature_names = list(split_data.stat_filter.kept_columns)
    else:
        ks_feature_names = list(split_data.feature_columns)

    # Se few-shot foi executado, usa o holdout (amostras não vistas no treino)
    if "few_shot_holdout" in pipeline_result:
        holdout = pipeline_result["few_shot_holdout"]
        logger.info("Usando holdout few-shot (%d amostras) para avaliação.", len(holdout["X"]))
        gen_result = test_generalization_from_arrays(
            pipeline=test_result.pipeline,
            X_gen=holdout["X"],
            y_true=holdout["y"],
            X_gen_normal=holdout["X_normal"],
            X_train_normal=X_train_normal,
            ks_feature_names=ks_feature_names,
            ks_threshold=0.1,
        )
    else:
        if not config.generalization_dataset.exists():
            logger.warning(
                "Dataset de generalização não encontrado: %s. Pulando.",
                config.generalization_dataset,
            )
            return pipeline_result

        gen_result = test_generalization(
            pipeline=test_result.pipeline,
            generalization_path=config.generalization_dataset,
            imputer=split_data.imputer,
            stat_filter=split_data.stat_filter,
            pca=split_data.pca,
            domain_features=config.domain_features,
            X_train_normal=X_train_normal,
            ks_feature_names=ks_feature_names,
            ks_threshold=0.1,
        )

    logger.info("\n%s", gen_result.report)
    logger.info(
        "Recall de ataque: %.2f%% | Drift detectado: %s",
        gen_result.recall_attack * 100,
        gen_result.drift_detected,
    )

    if gen_result.ks_result is not None:
        ks = gen_result.ks_result
        logger.info("-" * 50)
        logger.info("KS DRIFT POR FEATURE (top drifted):")
        top = sorted(ks.statistics.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, stat in top:
            marker = " ← DRIFT" if feat in ks.drifted_features else ""
            logger.info("  %-40s KS=%.3f%s", feat, stat, marker)
        logger.info(
            "Features com drift: %d/%d (%.1f%%)",
            len(ks.drifted_features),
            len(ks.statistics),
            ks.drift_ratio * 100,
        )

    pipeline_result["generalization"] = gen_result
    return pipeline_result


# ---------------------------------------------------------------------------
# Visualização PCA (opcional)
# ---------------------------------------------------------------------------
def run_pca_visualization(
    config: ExperimentConfig,
    pipeline_result: dict,
    *,
    n_components: int = 2,
    on: str = "test",
    sample_size: int = 10000,
    output_path: Path | None = None,
) -> dict:
    """
    Gera gráfico PCA 2D ou 3D com separação Normal vs Outlier.

    Args:
        config: Configurações do experimento.
        pipeline_result: Resultado de run_pipeline().
        n_components: 2 ou 3.
        on: Em qual conjunto plotar — "train", "val" ou "test". Para
            modo novelty, "train" só tem Normal e o gráfico fica monocromático;
            "test" é o mais informativo.
        sample_size: Pontos por classe (para legibilidade em datasets grandes).
        output_path: Caminho do PNG de saída.

    Returns:
        pipeline_result atualizado com a chave "pca_path".
    """
    from niod.visualization.pca_plot import generate_pca_plot

    split_data = pipeline_result["split"]

    if on == "train":
        X, y = split_data.X_train, split_data.y_train
    elif on == "val":
        X, y = split_data.X_val, split_data.y_val
    elif on == "test":
        X, y = split_data.X_test, split_data.y_test
    else:
        raise ValueError(f"--pca-on deve ser train/val/test, recebido: {on}")

    suffix = "filtered" if config.apply_filters else "raw"
    default_path = Path(f"pca_{n_components}d_{on}_{suffix}.png")
    output_path = output_path or default_path

    title = (
        f"PCA {n_components}D — {on} "
        f"({'com filtros' if config.apply_filters else 'sem filtros'}, "
        f"{config.algorithm.value})"
    )

    logger.info("=" * 70)
    logger.info("VISUALIZAÇÃO PCA (%dD)", n_components)
    logger.info("=" * 70)

    pca_path = generate_pca_plot(
        X=X,
        y=y,
        n_components=n_components,
        output_path=output_path,
        title=title,
        sample_size=sample_size,
        random_state=config.random_state,
    )

    pipeline_result["pca_path"] = pca_path
    return pipeline_result


def run_pca_cross_domain(
    config: ExperimentConfig,
    pipeline_result: dict,
    *,
    sample_size: int = 5000,
    output_path: Path | None = None,
    n_components: int = 2,
    interactive: bool = False,
) -> dict:
    """
    Gera PCA combinando o dataset de treino (ex: Friday) e o de
    generalização (ex: Tuesday), com 4 categorias: Normal/Ataque
    em cada dia. Útil para visualizar o concept drift.

    Reutiliza o stat_filter e o imputer do pipeline (sem refit) para
    garantir que ambos os domínios sejam projetados no mesmo espaço.
    """
    from niod.utils.data import (
        apply_imputation,
        clean_features,
        load_arff,
        transform_labels,
    )
    from niod.visualization.pca_plot import generate_pca_cross_domain

    split_data = pipeline_result["split"]

    if not config.generalization_dataset.exists():
        logger.warning(
            "Dataset de generalização não encontrado: %s. Pulando PCA cross-domain.",
            config.generalization_dataset,
        )
        return pipeline_result

    logger.info("=" * 70)
    logger.info("PCA CROSS-DOMAIN (TREINO vs GENERALIZAÇÃO)")
    logger.info("=" * 70)

    # ---- Treino (já no espaço filtrado/imputado) -------------------------
    X_train_full = np.asarray(split_data.X_train)
    y_train_full = np.asarray(split_data.y_train)
    X_test_full = np.asarray(split_data.X_test)
    y_test_full = np.asarray(split_data.y_test)

    # Em modo novelty, X_train é 100% normal; ataques estão só em val/teste.
    # Combinamos tudo e separamos por label.
    X_train_combined = np.vstack([X_train_full, X_test_full])
    y_train_combined = np.concatenate([y_train_full, y_test_full])
    X_train_normal = X_train_combined[y_train_combined == 0]
    X_train_attack = X_train_combined[y_train_combined == 1]

    if len(X_train_attack) == 0:
        logger.warning(
            "Nenhum ataque encontrado no dataset de treino — "
            "o gráfico cross-domain ficará incompleto."
        )

    # ---- Generalização (carregar e processar igual ao test_generalization) -
    logger.info("Carregando %s para PCA cross-domain...", config.generalization_dataset)
    df_gen = load_arff(config.generalization_dataset)

    # Aplica as mesmas features de domínio do treino (se houver)
    if config.domain_features:
        from niod.utils.domain_features import add_domain_features

        df_gen = add_domain_features(df_gen, features=config.domain_features)

    if "Label" not in df_gen.columns:
        logger.warning(
            "Coluna 'Label' não encontrada em %s. Pulando PCA cross-domain.",
            config.generalization_dataset,
        )
        return pipeline_result

    # Limpeza inicial: inf -> nan, e descarte de linhas TOTALMENTE inválidas
    # (mantém as parciais para o imputer tratar)
    df_gen = df_gen.replace([np.inf, -np.inf], np.nan)
    df_gen = df_gen.dropna(axis=0, how="all")

    if len(df_gen) == 0:
        logger.warning(
            "Dataset de generalização ficou vazio após limpeza. Pulando PCA cross-domain."
        )
        return pipeline_result

    X_gen_raw = df_gen.drop(columns=["Label"])
    y_gen = transform_labels(df_gen["Label"].values)  # 1 = Normal, -1 = Outlier

    # Pipeline idêntico ao test_generalization: clean → filter → impute → pca
    X_gen_clean = clean_features(X_gen_raw)
    if split_data.stat_filter is not None:
        X_gen_clean = split_data.stat_filter.transform(X_gen_clean)
    X_gen_clean = apply_imputation(X_gen_clean, split_data.imputer, fit=False)
    if split_data.pca is not None:
        X_gen_clean = split_data.pca.transform(X_gen_clean)

    X_gen_arr = np.asarray(X_gen_clean)
    X_gen_normal = X_gen_arr[y_gen == 1]
    X_gen_attack = X_gen_arr[y_gen == -1]

    logger.info(
        "Generalização processada: %d amostras (Normal=%d, Ataque=%d)",
        len(X_gen_arr),
        len(X_gen_normal),
        len(X_gen_attack),
    )

    # ---- Plot ------------------------------------------------------------
    suffix = "filtered" if config.apply_filters else "raw"
    train_name = config.train_dataset.stem
    gen_name = config.generalization_dataset.stem
    default_path = Path(
        f"pca_cross_{n_components}d_{train_name}_vs_{gen_name}_{suffix}.png"
    )
    output_path = output_path or default_path

    pca_path = generate_pca_cross_domain(
        X_train_normal=X_train_normal,
        X_train_attack=X_train_attack,
        X_gen_normal=X_gen_normal,
        X_gen_attack=X_gen_attack,
        train_label=train_name.replace("_", " ").title(),
        gen_label=gen_name.replace("_", " ").title(),
        output_path=output_path,
        sample_size=sample_size,
        random_state=config.random_state,
        n_components=n_components,
        interactive=interactive,
    )

    pipeline_result["pca_cross_domain_path"] = pca_path
    return pipeline_result


# ---------------------------------------------------------------------------
# Visualização UMAP (opcional)
# ---------------------------------------------------------------------------
def run_visualization(
    config: ExperimentConfig,
    pipeline_result: dict,
    umap_train_path: Path | None = None,
    umap_target_path: Path | None = None,
    *,
    interactive: bool = False,
) -> dict:
    """
    Gera visualização UMAP 3D para análise de drift.

    Args:
        config: Configurações do experimento.
        pipeline_result: Resultado de run_pipeline().
        umap_train_path: Dataset de treino para UMAP (default: data/Friday.arff).
        umap_target_path: Dataset alvo para UMAP (default: data/Tuesday.arff).

    Returns:
        Dicionário atualizado com caminho do gráfico.
    """
    from niod.visualization.umap_plot import generate_umap_3d

    test_result: EvaluationResult = pipeline_result["test"]
    split_data = pipeline_result["split"]

    umap_train = umap_train_path or Path("data/Friday.arff")
    umap_target = umap_target_path or Path("data/Tuesday.arff")

    if not umap_train.exists() or not umap_target.exists():
        logger.warning(
            "Datasets UMAP não encontrados (%s, %s). Pulando visualização.",
            umap_train,
            umap_target,
        )
        return pipeline_result

    # Recuperar transformer (pipeline sem o estimador final)
    trained_pipeline = test_result.pipeline
    transformer = trained_pipeline[:-1]

    # Calcular médias do treino para imputação consistente
    df_train = load_arff(config.train_dataset)
    df_train, _ = clean_dataframe(df_train)
    X_train_ref = df_train[df_train["Label"] == 0].drop(columns=["Label"])
    train_means = X_train_ref.replace([np.inf, -np.inf], np.nan).mean()

    output_path = generate_umap_3d(
        transformer=transformer,
        columns_ref=split_data.feature_columns,
        train_means=train_means,
        train_dataset_path=umap_train,
        target_dataset_path=umap_target,
        config=config,
        interactive=interactive,
    )

    pipeline_result["umap_path"] = output_path
    return pipeline_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_default_params(config: ExperimentConfig) -> dict:
    """Resolve os parâmetros default com base no algoritmo e modo."""
    algo = config.algorithm.value

    if algo in DEFAULT_PARAMS:
        params = DEFAULT_PARAMS[algo].copy()
    else:
        params = {}

    # Se não é novelty, usar contaminação do config
    if not config.novelty and "contamination" not in params:
        params["contamination"] = config.contamination

    return params


def _log_evaluation(stage: str, result: EvaluationResult) -> None:
    """Loga os resultados de uma avaliação de forma padronizada."""
    logger.info("")
    logger.info("=" * 50)
    logger.info("Resultado — %s", stage)
    logger.info("=" * 50)
    logger.info("Melhores parâmetros: %s", result.params)
    logger.info("F1 Score: %.4f", result.f1)
    logger.info("\nRelatório de Classificação:")
    logger.info("\n%s", result.report)
    logger.info("Matriz de Confusão [Normal, Outlier]:")
    logger.info("\n%s", result.confusion_matrix)


def _resolve_domain_features(args: argparse.Namespace) -> list[str] | None:
    """
    Resolve o valor final de domain_features a partir dos args do CLI.

    Prioriza --all-domain-features sobre --domain-features.
    Valida nomes contra o registro DOMAIN_FEATURES e aborta se houver
    nomes desconhecidos (falha rápida — melhor que silenciar).
    """
    from niod.utils.domain_features import DOMAIN_FEATURES

    if args.all_domain_features:
        return list(DOMAIN_FEATURES.keys())

    if not args.domain_features:
        return None

    # Validação: aborta se algum nome for desconhecido
    unknown = [n for n in args.domain_features if n not in DOMAIN_FEATURES]
    if unknown:
        available = ", ".join(DOMAIN_FEATURES.keys())
        raise SystemExit(
            f"Erro: domain features desconhecidas: {unknown}. "
            f"Disponíveis: {available}"
        )

    return args.domain_features


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="NIOD — Network Intrusion Outlier Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("data/Friday_balanceado.arff"),
        help="Caminho do dataset de treino (.arff)",
    )
    parser.add_argument(
        "--generalization-dataset",
        type=Path,
        default=Path("data/Tuesday.arff"),
        help="Caminho do dataset de generalização (.arff)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[a.value for a in Algorithm],
        default=Algorithm.ISOLATION_FOREST.value,
        help="Algoritmo de detecção",
    )
    parser.add_argument(
        "--no-novelty",
        action="store_true",
        help="Desativar modo novelty (inclui outliers no treino)",
    )
    parser.add_argument(
        "--no-hyper-search",
        action="store_true",
        help="Desativar busca de hiperparâmetros",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Taxa de contaminação (se novelty=False)",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Pular geração de UMAP 3D",
    )
    parser.add_argument(
        "--umap-interactive",
        action="store_true",
        help="Salvar UMAP 3D como HTML interativo (rotacionável no browser)",
    )
    parser.add_argument(
        "--skip-generalization",
        action="store_true",
        help="Pular teste de generalização",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Pular geração do gráfico PCA",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Número de componentes do PCA (2 ou 3)",
    )
    parser.add_argument(
        "--pca-on",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Conjunto onde aplicar PCA para plotar",
    )
    parser.add_argument(
        "--pca-sample-size",
        type=int,
        default=10000,
        help="Pontos por classe no gráfico PCA (subamostragem)",
    )
    parser.add_argument(
        "--skip-pca-cross",
        action="store_true",
        help="Pular geração do PCA cross-domain (treino vs generalização)",
    )
    parser.add_argument(
        "--pca-cross-sample-size",
        type=int,
        default=5000,
        help="Pontos por categoria no PCA cross-domain (subamostragem)",
    )
    parser.add_argument(
        "--pca-cross-components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Número de componentes do PCA cross-domain (2 ou 3)",
    )
    parser.add_argument(
        "--pca-cross-interactive",
        action="store_true",
        help="Salvar PCA cross-domain como HTML interativo (Plotly, rotacionável no browser). "
        "Recomendado para 3D, onde o ângulo de câmera estático pode esconder estrutura.",
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Desativar filtros estatísticos (variância/duplicatas/correlação)",
    )
    parser.add_argument(
        "--pca-reduce",
        type=int,
        default=None,
        metavar="N",
        help="Aplicar PCA com N componentes como redução de dimensionalidade "
        "(fitado só no treino, propagado a val/teste/generalização). "
        "Padrão: desativado.",
    )
    parser.add_argument(
        "--no-pca-reduce",
        action="store_true",
        help="Desativa PCA como redução de dimensionalidade (sobrescreve --pca-reduce).",
    )
    parser.add_argument(
        "--domain-features",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Lista de features de domínio a adicionar. Nomes disponíveis: "
        "Eng_Packet_Shape, Eng_Fwd_Header_Load, Eng_Temporal_Burstiness, "
        "Eng_Flag_Density, Eng_Flow_Indicators. "
        "Ex: --domain-features Eng_Packet_Shape Eng_Flag_Density",
    )
    parser.add_argument(
        "--all-domain-features",
        action="store_true",
        help="Ativa TODAS as features de domínio registradas (atalho).",
    )
    parser.add_argument(
        "--extra-normal-dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Dataset extra para enriquecer o treino com amostras normais adicionais "
        "(ex: data/Tuesday.arff). Só normais são extraídas; pré-processamento do "
        "treino é reutilizado sem refitting.",
    )
    parser.add_argument(
        "--few-shot-dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Dataset alvo para few-shot não-supervisionado (ex: data/Tuesday.arff). "
        "Uma fração das normais do alvo é adicionada ao treino; os ataques (+ normais "
        "restantes) formam o holdout de avaliação de generalização.",
    )
    parser.add_argument(
        "--few-shot-ratio",
        type=float,
        default=0.05,
        metavar="RATIO",
        help="Fração das amostras normais do dataset alvo a incluir no treino "
        "(padrão: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de logging",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point da CLI."""
    args = parse_args()

    pca_reduce_value = None if args.no_pca_reduce else args.pca_reduce
    domain_features_value = _resolve_domain_features(args)

    config = ExperimentConfig(
        train_dataset=args.train_dataset,
        generalization_dataset=args.generalization_dataset,
        algorithm=Algorithm(args.algorithm),
        novelty=not args.no_novelty,
        hyper_search=not args.no_hyper_search,
        contamination=args.contamination,
        apply_filters=not args.no_filters,
        pca_reduce=pca_reduce_value,
        domain_features=domain_features_value,
        extra_normal_dataset=args.extra_normal_dataset,
        few_shot_dataset=args.few_shot_dataset,
        few_shot_ratio=args.few_shot_ratio,
        log_level=args.log_level,
    )

    # 1. Pipeline principal
    results = run_pipeline(config)

    # 1b. Enriquecimento do treino com normais extras (opcional)
    if config.extra_normal_dataset is not None:
        results = enrich_train_with_extra_normal(config, results)

    # 1c. Few-shot: adapta modelo ao domínio alvo (opcional)
    if config.few_shot_dataset is not None:
        results = run_few_shot_enrichment(config, results)

    # 2. Generalização (opcional) — usa holdout few-shot se disponível
    if not args.skip_generalization:
        results = run_generalization(config, results)

    # 3. PCA (opcional)
    if not args.skip_pca:
        results = run_pca_visualization(
            config,
            results,
            n_components=args.pca_components,
            on=args.pca_on,
            sample_size=args.pca_sample_size,
        )

    # 3b. PCA cross-domain (treino vs generalização)
    if not args.skip_pca_cross:
        results = run_pca_cross_domain(
            config,
            results,
            sample_size=args.pca_cross_sample_size,
            n_components=args.pca_cross_components,
            interactive=args.pca_cross_interactive,
        )

    # 4. Visualização UMAP (opcional)
    if not args.skip_umap:
        results = run_visualization(
            config, results, interactive=args.umap_interactive
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline concluído com sucesso.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
