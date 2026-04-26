"""
NIOD — Pipeline principal de detecção de anomalias em tráfego de rede.

Orquestra todo o fluxo: carregamento → split → treino → avaliação →
busca de hiperparâmetros → teste de generalização → visualização UMAP.

Uso:
    python -m niod.main
    python -m niod.main --algorithm svm --no-novelty --no-hyper-search
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
    from niod.modules.generalization import test_generalization

    test_result: EvaluationResult = pipeline_result["test"]
    split_data = pipeline_result["split"]

    if not config.generalization_dataset.exists():
        logger.warning(
            "Dataset de generalização não encontrado: %s. Pulando.",
            config.generalization_dataset,
        )
        return pipeline_result

    logger.info("=" * 70)
    logger.info("TESTE DE GENERALIZAÇÃO (CONCEPT DRIFT)")
    logger.info("=" * 70)

    gen_result = test_generalization(
        pipeline=test_result.pipeline,
        generalization_path=config.generalization_dataset,
        imputer=split_data.imputer,
        stat_filter=split_data.stat_filter,
        pca=split_data.pca,
        domain_features=config.domain_features,
    )

    logger.info("\n%s", gen_result.report)
    logger.info(
        "Recall de ataque: %.2f%% | Drift detectado: %s",
        gen_result.recall_attack * 100,
        gen_result.drift_detected,
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
    default_path = Path(f"pca_cross_{train_name}_vs_{gen_name}_{suffix}.png")
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
        "Eng_Packet_Shape, Eng_Fwd_Header_Load. Ex: --domain-features Eng_Packet_Shape",
    )
    parser.add_argument(
        "--all-domain-features",
        action="store_true",
        help="Ativa TODAS as features de domínio registradas (atalho).",
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

    if args.all_domain_features:
        from niod.utils.domain_features import DOMAIN_FEATURES

        domain_features_value = list(DOMAIN_FEATURES.keys())
    else:
        domain_features_value = args.domain_features  # None ou lista

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
        log_level=args.log_level,
    )

    # 1. Pipeline principal
    results = run_pipeline(config)

    # 2. Generalização (opcional)
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
        )

    # 4. Visualização UMAP (opcional)
    if not args.skip_umap:
        results = run_visualization(config, results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline concluído com sucesso.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
