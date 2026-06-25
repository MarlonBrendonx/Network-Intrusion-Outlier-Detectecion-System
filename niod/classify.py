from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from niod.config.settings import ClassificationAlgorithm, ExperimentConfig
from niod.modules.classification import (
    ClassificationResult,
    build_classifier,
    evaluate_classifier,
    get_classifier_factory,
    hyperparameters_search_classifier,
)
from niod.utils.data import clean_dataframe, load_arff, prepare_splits

logger = logging.getLogger("niod")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("joblib").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)


def run_classification_pipeline(
    train_dataset: Path,
    algorithm: str = "xgboost",
    hyper_search: bool = False,
    apply_filters: bool = True,
    domain_features: list[str] | None = None,
    pca_reduce: int | None = None,
    use_smote: bool = True,
    temporal_split: bool = False,
    log_level: str = "INFO",
    random_state: int = 42,
) -> dict:
    _setup_logging(log_level)
    results: dict = {}

    logger.info("Algorithm:    %s", algorithm)
    logger.info("Hyper Search: %s", hyper_search)
    logger.info("Filters:      %s", apply_filters)
    logger.info("PCA reduce:   %s", pca_reduce or "off")
    logger.info("Domain feat:  %s", domain_features or "off")
    logger.info(
        "SMOTE:        %s (applied only on the training set, post-split)",
        "on" if use_smote else "off",
    )
    logger.info(
        "Split:        %s",
        "POSITIONAL (temporal proxy)" if temporal_split else "random (stratified)",
    )
    logger.info("Dataset:      %s", train_dataset)

    df = load_arff(train_dataset)
    df, removed = clean_dataframe(df)
    logger.info("Final shape: %s (%d rows removed)", df.shape, removed)

    logger.info("-" * 70)
    logger.info("Splitting data (stratified, both classes in the training set)...")

    split_data = prepare_splits(
        df,
        label_column="Label",
        novelty=False,
        contamination=None,
        train_size=0.6,
        random_state=random_state,
        apply_filters=apply_filters,
        pca_reduce=pca_reduce,
        domain_features=domain_features,
        temporal_split=temporal_split,
    )
    results["split"] = split_data

    y_train = split_data.y_train
    y_val = split_data.y_val
    y_test = split_data.y_test

    logger.info(
        "Attack proportion — Train: %.4f | Val: %.4f | Test: %.4f",
        np.mean(y_train),
        np.mean(y_val),
        np.mean(y_test),
    )

    model_factory = get_classifier_factory(algorithm)

    if hyper_search:
        logger.info("-" * 70)
        logger.info("Starting hyperparameter search (%s)...", algorithm)

        val_result = hyperparameters_search_classifier(
            algorithm,
            split_data.X_train,
            y_train,
            split_data.X_val,
            y_val,
            use_smote=use_smote,
            random_state=random_state,
        )
        _log_result("VALIDATION", val_result)
        results["validation"] = val_result

        logger.info("-" * 70)
        logger.info("Final evaluation on the TEST set...")

        test_result = evaluate_classifier(
            model_factory,
            val_result.params,
            split_data.X_train,
            y_train,
            split_data.X_test,
            y_test,
            use_smote=use_smote,
            random_state=random_state,
        )
    else:
        logger.info("Training with default parameters...")
        test_result = evaluate_classifier(
            model_factory,
            {},
            split_data.X_train,
            y_train,
            split_data.X_test,
            y_test,
            use_smote=use_smote,
            random_state=random_state,
        )

    train_result = evaluate_classifier(
        model_factory,
        test_result.params,
        split_data.X_train,
        y_train,
        split_data.X_train,
        y_train,
        verbose=False,
        use_smote=use_smote,
        random_state=random_state,
    )
    logger.info("")
    logger.info(
        "F1 TRAIN:  %.4f | F1 TEST: %.4f | Gap: %.4f",
        train_result.f1,
        test_result.f1,
        train_result.f1 - test_result.f1,
    )

    _log_result("TEST", test_result)
    results["train"] = train_result
    results["test"] = test_result
    return results


def run_few_shot_enrichment(
    few_shot_dataset: Path,
    results: dict,
    *,
    ratio: float = 0.05,
    domain_features: list[str] | None = None,
    use_smote: bool = True,
    random_state: int = 42,
) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.utils import shuffle as sk_shuffle

    from niod.utils.data import apply_imputation, clean_features, load_arff
    from niod.utils.domain_features import add_domain_features

    if not few_shot_dataset.exists():
        logger.warning(
            "Few-shot dataset not found: %s. Skipping.", few_shot_dataset
        )
        return results

    logger.info("=" * 70)
    logger.info("FEW-SHOT ENRICHMENT (ratio=%.0f%%)", ratio * 100)
    logger.info("=" * 70)
    logger.info("Source: %s", few_shot_dataset)

    split_data = results["split"]

    df = load_arff(few_shot_dataset)
    if domain_features:
        df = add_domain_features(df, features=domain_features)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    df_attacks = df[df["Label"] == 1]
    df_normal = df[df["Label"] == 0]

    n_few_shot = max(1, int(len(df_attacks) * ratio))
    df_attacks_shuffled = df_attacks.sample(frac=1, random_state=random_state)
    df_few = df_attacks_shuffled.iloc[:n_few_shot]
    df_holdout_attacks = df_attacks_shuffled.iloc[n_few_shot:]

    df_holdout = pd.concat([df_normal, df_holdout_attacks])

    logger.info(
        "Target attacks: %d total → %d few-shot | %d holdout",
        len(df_attacks),
        len(df_few),
        len(df_holdout_attacks),
    )

    def _preprocess(df_subset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y = df_subset["Label"].values.astype(int)
        X = df_subset.drop(columns=["Label"])
        X = clean_features(X)
        if split_data.stat_filter is not None:
            X = split_data.stat_filter.transform(X)
        X = apply_imputation(X, split_data.imputer, fit=False)
        if split_data.pca is not None:
            X = split_data.pca.transform(X)
        return np.asarray(X), y

    X_few, y_few = _preprocess(df_few)

    X_train_enriched = np.vstack([split_data.X_train, X_few])
    y_train_enriched = np.concatenate([split_data.y_train, y_few])

    X_train_enriched, y_train_enriched = sk_shuffle(
        X_train_enriched, y_train_enriched, random_state=random_state
    )

    logger.info(
        "X_train: %d → %d samples (+%d attacks from %s)",
        len(split_data.X_train),
        len(X_train_enriched),
        len(X_few),
        few_shot_dataset.stem,
    )

    old_result: ClassificationResult = results["test"]
    model_factory = get_classifier_factory("xgboost")
    new_model = build_classifier(
        model_factory,
        old_result.params,
        use_smote=use_smote,
        random_state=random_state,
    )
    new_model.fit(X_train_enriched, y_train_enriched)

    y_test_pred = new_model.predict(split_data.X_test)
    y_test = split_data.y_test
    f1_test = f1_score(y_test, y_test_pred, labels=[0, 1], average="macro")
    logger.info(
        "F1 on the Friday test set after few-shot: %.4f (before: %.4f)", f1_test, old_result.f1
    )

    results["test"] = ClassificationResult(
        f1=f1_test,
        params=old_result.params,
        report=classification_report(
            y_test,
            y_test_pred,
            labels=[0, 1],
            target_names=["Normal (0)", "Attack (1)"],
        ),
        confusion_matrix=confusion_matrix(y_test, y_test_pred, labels=[0, 1]),
        model=new_model,
    )

    X_holdout, y_holdout = _preprocess(df_holdout)
    results["few_shot_holdout"] = {"X": X_holdout, "y": y_holdout}

    return results


def _log_result(stage: str, result: ClassificationResult) -> None:
    logger.info("")
    logger.info("=" * 50)
    logger.info("Result — %s", stage)
    logger.info("=" * 50)
    logger.info("Parameters: %s", result.params)
    logger.info("F1 Score:   %.4f", result.f1)
    logger.info("\nClassification Report:")
    logger.info("\n%s", result.report)
    logger.info("Confusion Matrix [Normal, Attack]:")
    logger.info("\n%s", result.confusion_matrix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NIOD — Supervised Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("data/Friday.arff"),
        help="Path to the training dataset (.arff). Use the RAW (imbalanced) dataset: "
        "SMOTE is applied internally only on the training set, after the split.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[a.value for a in ClassificationAlgorithm],
        default=ClassificationAlgorithm.XGBOOST.value,
        help="Classification algorithm",
    )
    parser.add_argument(
        "--hyper-search",
        action="store_true",
        help="Enable hyperparameter search (disabled by default)",
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Disable statistical filters",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE (train on the real imbalance). By default SMOTE "
        "is applied ONLY on the training set, after the split — val/test remain real.",
    )
    parser.add_argument(
        "--pca-reduce",
        type=int,
        default=None,
        metavar="N",
        help="Apply PCA with N components before training",
    )
    parser.add_argument(
        "--domain-features",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Domain features to add (e.g.: Eng_Flag_Density Eng_Flow_Rates)",
    )
    parser.add_argument(
        "--all-domain-features",
        action="store_true",
        help="Enable ALL registered domain features",
    )
    parser.add_argument(
        "--few-shot-dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Target dataset for few-shot (e.g.: data/Tuesday.arff). A fraction "
        "of the attacks is added to the training set; the rest becomes the evaluation holdout.",
    )
    parser.add_argument(
        "--few-shot-ratio",
        type=float,
        default=0.05,
        metavar="RATIO",
        help="Fraction of the target dataset's attacks to include in the training set (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--temporal-split",
        action="store_true",
        help="Use a positional split (temporal proxy) instead of a random one: train=start, "
        "val=middle, test=end, preserving the row order. Prevents twin flows from the "
        "same attack episode from leaking between train and test. Without a timestamp in the ARFF, it is the "
        "best available temporal proxy.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    domain_features = None
    if args.all_domain_features:
        from niod.utils.domain_features import DOMAIN_FEATURES

        domain_features = list(DOMAIN_FEATURES.keys())
    elif args.domain_features:
        domain_features = args.domain_features

    results = run_classification_pipeline(
        train_dataset=args.train_dataset,
        algorithm=args.algorithm,
        hyper_search=args.hyper_search,
        apply_filters=not args.no_filters,
        domain_features=domain_features,
        pca_reduce=args.pca_reduce,
        use_smote=not args.no_smote,
        temporal_split=args.temporal_split,
        log_level=args.log_level,
    )

    if args.few_shot_dataset is not None:
        results = run_few_shot_enrichment(
            few_shot_dataset=args.few_shot_dataset,
            results=results,
            ratio=args.few_shot_ratio,
            domain_features=domain_features,
            use_smote=not args.no_smote,
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline completed successfully.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
