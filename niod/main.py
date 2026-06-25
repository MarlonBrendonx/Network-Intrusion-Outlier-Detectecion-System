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

from niod.utils.data import load_extra_normal
from niod.visualization.pca_plot import generate_pca_plot

from niod.utils.domain_features import FEATURE_TO_GROUP
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from niod.utils.data import (
    apply_imputation,
    clean_features,
    load_arff,
    transform_labels,
)
from niod.utils.domain_features import add_domain_features

logger = logging.getLogger("niod")


def _setup_logging(level: str = "INFO") -> None:
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
    )
    logging.getLogger("joblib").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


def run_pipeline(config: ExperimentConfig) -> dict:
    _setup_logging(config.log_level)
    results: dict = {"config": config}

    logger.info("=" * 70)
    logger.info("=" * 70)
    logger.info("Algorithm:   %s", config.algorithm.value)
    logger.info("Novelty:     %s", config.novelty)
    logger.info("Hyper Search: %s", config.hyper_search)
    logger.info("Filters:     %s", config.apply_filters)
    logger.info("PCA reduce:  %s", config.pca_reduce or "off")
    logger.info("Domain feat: %s", config.domain_features or "off")
    logger.info("Dataset:     %s", config.train_dataset)

    df = load_arff(config.train_dataset)
    df, removed = clean_dataframe(df)
    logger.info("Final shape: %s (removed %d rows)", df.shape, removed)

    logger.info("-" * 70)
    logger.info("Splitting data...")
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
        feature_whitelist=config.feature_whitelist,
    )
    results["split"] = split_data

    algorithm_name = config.algorithm.value
    model_factory = get_model_factory(algorithm_name)

    if config.hyper_search:
        logger.info("-" * 70)
        logger.info("Starting hyperparameter search (%s)...", algorithm_name)

        val_result = hyperparameters_search(
            algorithm_name,
            split_data.X_train,
            split_data.X_val,
            split_data.y_val_transformed,
        )

        _log_evaluation("VALIDATION", val_result)
        results["validation"] = val_result

        logger.info("-" * 70)
        logger.info("Final evaluation on the TEST set...")

        best_params = val_result.params
        test_result = evaluate_model(
            model_factory,
            best_params,
            split_data.X_train,
            split_data.X_test,
            split_data.y_test_transformed,
        )
    else:
        params = _resolve_default_params(config)
        logger.info("Fixed parameters: %s", params)

        # Evaluate on VALIDATION even without a grid: modeling decisions (PCA, domain
        # features) should be made on the validation set, not the test set. Without this,
        # the only number printed was the test one, inducing selection on the test
        # set (review item 2). The test set is still evaluated only once.
        val_result = evaluate_model(
            model_factory,
            params,
            split_data.X_train,
            split_data.X_val,
            split_data.y_val_transformed,
        )
        _log_evaluation("VALIDATION", val_result)
        results["validation"] = val_result

        test_result = evaluate_model(
            model_factory,
            params,
            split_data.X_train,
            split_data.X_test,
            split_data.y_test_transformed,
        )

    _log_evaluation("TEST", test_result)
    results["test"] = test_result

    return results


def enrich_train_with_extra_normal(
    config: ExperimentConfig,
    pipeline_result: dict,
) -> dict:
    split_data = pipeline_result["split"]
    extra_path = config.extra_normal_dataset

    if extra_path is None or not extra_path.exists():
        logger.warning(
            "Extra dataset not found: %s. Skipping enrichment.", extra_path
        )
        return pipeline_result

    logger.info("=" * 70)
    logger.info("TRAINING ENRICHMENT WITH EXTRA NORMAL SAMPLES")
    logger.info("=" * 70)
    logger.info("Source: %s", extra_path)

    X_extra = load_extra_normal(
        extra_path=extra_path,
        stat_filter=split_data.stat_filter,
        imputer=split_data.imputer,
        feature_columns=split_data.feature_columns,
        domain_features=config.domain_features,
        pca=split_data.pca,
    )

    if len(X_extra) == 0:
        logger.warning("No extra samples added.")
        return pipeline_result

    original_size = len(split_data.X_train)
    split_data.X_train = np.vstack([split_data.X_train, X_extra])
    split_data.y_train = np.concatenate(
        [split_data.y_train, np.zeros(len(X_extra), dtype=split_data.y_train.dtype)]
    )
    logger.info(
        "X_train: %d → %d samples (+%d normal from %s)",
        original_size,
        len(split_data.X_train),
        len(X_extra),
        extra_path.stem,
    )

    return pipeline_result


def run_few_shot_enrichment(
    config: ExperimentConfig,
    pipeline_result: dict,
) -> dict:
    few_shot_path = config.few_shot_dataset
    ratio = config.few_shot_ratio

    if few_shot_path is None or not few_shot_path.exists():
        logger.warning("Few-shot dataset not found: %s. Skipping.", few_shot_path)
        return pipeline_result

    logger.info("=" * 70)
    logger.info("FEW-SHOT ENRICHMENT — UNSUPERVISED (ratio=%.0f%%)", ratio * 100)
    logger.info("=" * 70)
    logger.info("Source: %s", few_shot_path)

    split_data = pipeline_result["split"]
    test_result: EvaluationResult = pipeline_result["test"]

    df = load_arff(few_shot_path)
    if config.domain_features:
        df = add_domain_features(df, features=config.domain_features)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    df_normal = df[df["Label"] == 0]
    df_attacks = df[df["Label"] == 1]

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
        "Target normal samples: %d total → %d few-shot | %d holdout-normal | %d holdout-attacks",
        len(df_normal),
        len(X_few),
        len(df_holdout_normal),
        len(df_attacks),
    )

    X_train_enriched = np.vstack([split_data.X_train, X_few])
    split_data.X_train = X_train_enriched
    split_data.y_train = np.concatenate(
        [split_data.y_train, np.zeros(len(X_few), dtype=split_data.y_train.dtype)]
    )

    logger.info(
        "X_train: %d → %d samples (+%d normal from %s)",
        len(X_train_enriched) - len(X_few),
        len(X_train_enriched),
        len(X_few),
        few_shot_path.stem,
    )

    estimator = test_result.pipeline.named_steps["estimator"]
    is_lof = estimator.__class__.__name__ == "LocalOutlierFactor"
    is_novelty = getattr(estimator, "novelty", False)

    if is_lof and not is_novelty:
        logger.warning("LOF non-novelty mode: retraining ignored (fit_predict on test).")
    else:
        model_factory = get_model_factory(config.algorithm.value)
        new_pipeline = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("estimator", model_factory(**test_result.params)),
            ]
        )
        new_pipeline.fit(X_train_enriched)

        y_pred = new_pipeline.predict(split_data.X_test)
        f1_new = f1_score(
            split_data.y_test_transformed, y_pred, labels=[1, -1], average="macro"
        )
        logger.info(
            "F1 on the original test set after few-shot: %.4f (before: %.4f)",
            f1_new,
            test_result.f1,
        )

        pipeline_result["test"] = EvaluationResult(
            f1=f1_new,
            params=test_result.params,
            report=classification_report(
                split_data.y_test_transformed,
                y_pred,
                labels=[1, -1],
                target_names=["Normal (1)", "Outlier (-1)"],
            ),
            confusion_matrix=confusion_matrix(
                split_data.y_test_transformed, y_pred, labels=[1, -1]
            ),
            pipeline=new_pipeline,
        )

    pipeline_result["few_shot_holdout"] = {
        "X": X_holdout,
        "y": y_holdout,
        "X_normal": X_holdout_normal,
    }

    return pipeline_result


def run_pca_visualization(
    config: ExperimentConfig,
    pipeline_result: dict,
    *,
    n_components: int = 2,
    on: str = "test",
    sample_size: int = 10000,
    output_path: Path | None = None,
) -> dict:
    split_data = pipeline_result["split"]

    if on == "train":
        X, y = split_data.X_train, split_data.y_train
    elif on == "val":
        X, y = split_data.X_val, split_data.y_val
    elif on == "test":
        X, y = split_data.X_test, split_data.y_test
    else:
        raise ValueError(f"--pca-on must be train/val/test, received: {on}")

    suffix = "filtered" if config.apply_filters else "raw"
    default_path = Path(f"pca_{n_components}d_{on}_{suffix}.png")
    output_path = output_path or default_path

    title = (
        f"PCA {n_components}D — {on} "
        f"({'with filters' if config.apply_filters else 'without filters'}, "
        f"{config.algorithm.value})"
    )

    logger.info("=" * 70)
    logger.info("PCA VISUALIZATION (%dD)", n_components)
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
            "Generalization dataset not found: %s. Skipping cross-domain PCA.",
            config.generalization_dataset,
        )
        return pipeline_result

    X_train_full = np.asarray(split_data.X_train)
    y_train_full = np.asarray(split_data.y_train)
    X_test_full = np.asarray(split_data.X_test)
    y_test_full = np.asarray(split_data.y_test)

    X_train_combined = np.vstack([X_train_full, X_test_full])
    y_train_combined = np.concatenate([y_train_full, y_test_full])
    X_train_normal = X_train_combined[y_train_combined == 0]
    X_train_attack = X_train_combined[y_train_combined == 1]

    if len(X_train_attack) == 0:
        logger.warning(
            "No attacks found in the training dataset — "
            "the cross-domain plot will be incomplete."
        )

    logger.info("Loading %s for cross-domain PCA...", config.generalization_dataset)
    df_gen = load_arff(config.generalization_dataset)

    if config.domain_features:
        from niod.utils.domain_features import add_domain_features

        df_gen = add_domain_features(df_gen, features=config.domain_features)

    if "Label" not in df_gen.columns:
        logger.warning(
            "Column 'Label' not found in %s. Skipping cross-domain PCA.",
            config.generalization_dataset,
        )
        return pipeline_result

    df_gen = df_gen.replace([np.inf, -np.inf], np.nan)
    df_gen = df_gen.dropna(axis=0, how="all")

    if len(df_gen) == 0:
        logger.warning(
            "Generalization dataset became empty after cleaning. Skipping cross-domain PCA."
        )
        return pipeline_result

    X_gen_raw = df_gen.drop(columns=["Label"])
    y_gen = transform_labels(df_gen["Label"].values)

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
        "Generalization processed: %d samples (Normal=%d, Attack=%d)",
        len(X_gen_arr),
        len(X_gen_normal),
        len(X_gen_attack),
    )

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


def _resolve_default_params(config: ExperimentConfig) -> dict:
    algo = config.algorithm.value

    if algo in DEFAULT_PARAMS:
        params = DEFAULT_PARAMS[algo].copy()
    else:
        params = {}

    if not config.novelty and "contamination" not in params:
        params["contamination"] = config.contamination

    return params


def _log_evaluation(stage: str, result: EvaluationResult) -> None:
    logger.info("")
    logger.info("=" * 50)
    logger.info("Result — %s", stage)
    logger.info("=" * 50)
    logger.info("Best parameters: %s", result.params)
    logger.info("F1 Score: %.4f", result.f1)
    logger.info("\nClassification Report:")
    logger.info("\n%s", result.report)
    logger.info("Confusion Matrix [Normal, Outlier]:")
    logger.info("\n%s", result.confusion_matrix)


def _resolve_domain_features(args: argparse.Namespace) -> list[str] | None:
    from niod.utils.domain_features import DOMAIN_FEATURES, FEATURE_TO_GROUP

    if args.all_domain_features:
        return list(DOMAIN_FEATURES.keys())

    if not args.domain_features:
        return None

    valid = set(DOMAIN_FEATURES) | set(FEATURE_TO_GROUP)
    unknown = [n for n in args.domain_features if n not in valid]
    if unknown:
        groups = ", ".join(DOMAIN_FEATURES.keys())
        feats = ", ".join(FEATURE_TO_GROUP.keys())
        raise SystemExit(
            f"Error: unknown names: {unknown}.\n"
            f"Available groups: {groups}\n"
            f"Available features: {feats}"
        )

    return args.domain_features


def _augment_domain_for_whitelist(
    domain_features: list[str] | None,
    whitelist: list[str] | None,
) -> list[str] | None:
    if not whitelist:
        return domain_features

    needed_groups = {
        FEATURE_TO_GROUP[col] for col in whitelist if col in FEATURE_TO_GROUP
    }
    if not needed_groups:
        return domain_features

    merged = list(domain_features or [])
    added = [g for g in sorted(needed_groups) if g not in merged]
    if added:
        merged.extend(added)
        logger.info(
            "Whitelist requires domain features; adding groups: %s", added
        )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NIOD — Network Intrusion Outlier Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("data/Friday.arff"),
        help="Path to the training dataset (.arff)",
    )
    parser.add_argument(
        "--generalization-dataset",
        type=Path,
        default=Path("data/Tuesday.arff"),
        help="Path to the generalization dataset (.arff)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[a.value for a in Algorithm],
        default=Algorithm.ISOLATION_FOREST.value,
        help="Detection algorithm",
    )
    parser.add_argument(
        "--no-novelty",
        action="store_true",
        help="Disable novelty mode (includes outliers in training)",
    )
    parser.add_argument(
        "--hyper-search",
        action="store_true",
        help="Enable hyperparameter search (disabled by default)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Contamination rate (if novelty=False)",
    )
    parser.add_argument(
        "--plot-pca",
        action="store_true",
        help="Generate the PCA plot (disabled by default)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of PCA components (2 or 3)",
    )
    parser.add_argument(
        "--pca-on",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Set on which to apply PCA for plotting",
    )
    parser.add_argument(
        "--pca-sample-size",
        type=int,
        default=10000,
        help="Points per class in the PCA plot (subsampling)",
    )
    parser.add_argument(
        "--plot-pca-cross",
        action="store_true",
        help="Generate the cross-domain PCA (training vs generalization; disabled by default)",
    )
    parser.add_argument(
        "--pca-cross-sample-size",
        type=int,
        default=5000,
        help="Points per category in the cross-domain PCA (subsampling)",
    )
    parser.add_argument(
        "--pca-cross-components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of cross-domain PCA components (2 or 3)",
    )
    parser.add_argument(
        "--pca-cross-interactive",
        action="store_true",
        help="Save the cross-domain PCA as interactive HTML (Plotly, rotatable in the browser). "
        "Recommended for 3D, where a static camera angle can hide structure.",
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Disable statistical filters (variance/duplicates/correlation)",
    )
    parser.add_argument(
        "--pca-reduce",
        type=int,
        default=None,
        metavar="N",
        help="Apply PCA with N components as dimensionality reduction "
        "(fitted only on training, propagated to val/test/generalization). "
        "Default: disabled.",
    )
    parser.add_argument(
        "--no-pca-reduce",
        action="store_true",
        help="Disable PCA as dimensionality reduction (overrides --pca-reduce).",
    )
    parser.add_argument(
        "--domain-features",
        nargs="+",
        default=None,
        metavar="NAME",
        help="List of GROUPS or individual FEATURES to add (can be mixed). "
        "Groups: Eng_Packet_Shape, Eng_Fwd_Header_Load, Eng_Temporal_Burstiness, "
        "Eng_Flag_Density, Eng_Flow_Indicators, Eng_Flow_Rates. "
        "Ex (group): --domain-features Eng_Flag_Density | "
        "Ex (features): --domain-features is_short_flow is_unidirectional",
    )
    parser.add_argument(
        "--all-domain-features",
        action="store_true",
        help="Enable ALL registered domain features (shortcut).",
    )
    parser.add_argument(
        "--feature-whitelist",
        nargs="+",
        default=None,
        metavar="COL",
        help="Train using ONLY these columns (base and/or domain), discarding "
        "all others. Applied after generating the domain features; disables the "
        "statistical filters (the whitelist takes priority). Use quotes for names with "
        'spaces. Ex: --feature-whitelist fwd_header_to_payload_ratio "ACK Flag Count"',
    )
    parser.add_argument(
        "--extra-normal-dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Extra dataset to enrich training with additional normal samples "
        "(e.g. data/Tuesday.arff). Only normal samples are extracted; the training "
        "preprocessing is reused without refitting.",
    )
    parser.add_argument(
        "--few-shot-dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Target dataset for unsupervised few-shot (e.g. data/Tuesday.arff). "
        "A fraction of the target's normal samples is added to training; the attacks (+ remaining "
        "normal samples) form the generalization evaluation holdout.",
    )
    parser.add_argument(
        "--few-shot-ratio",
        type=float,
        default=0.05,
        metavar="RATIO",
        help="Fraction of the target dataset's normal samples to include in training "
        "(default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pca_reduce_value = None if args.no_pca_reduce else args.pca_reduce
    domain_features_value = _resolve_domain_features(args)
    domain_features_value = _augment_domain_for_whitelist(
        domain_features_value, args.feature_whitelist
    )

    config = ExperimentConfig(
        train_dataset=args.train_dataset,
        generalization_dataset=args.generalization_dataset,
        algorithm=Algorithm(args.algorithm),
        novelty=not args.no_novelty,
        hyper_search=args.hyper_search,
        contamination=args.contamination,
        apply_filters=not args.no_filters,
        pca_reduce=pca_reduce_value,
        domain_features=domain_features_value,
        feature_whitelist=args.feature_whitelist,
        extra_normal_dataset=args.extra_normal_dataset,
        few_shot_dataset=args.few_shot_dataset,
        few_shot_ratio=args.few_shot_ratio,
        log_level=args.log_level,
    )

    results = run_pipeline(config)

    if config.extra_normal_dataset is not None:
        results = enrich_train_with_extra_normal(config, results)

    if config.few_shot_dataset is not None:
        results = run_few_shot_enrichment(config, results)

    if args.plot_pca:
        results = run_pca_visualization(
            config,
            results,
            n_components=args.pca_components,
            on=args.pca_on,
            sample_size=args.pca_sample_size,
        )

    if args.plot_pca_cross:
        results = run_pca_cross_domain(
            config,
            results,
            sample_size=args.pca_cross_sample_size,
            n_components=args.pca_cross_components,
            interactive=args.pca_cross_interactive,
        )


if __name__ == "__main__":
    main()
