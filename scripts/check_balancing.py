from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from niod.utils.data import clean_dataframe, load_arff

LABEL_COLUMN = "Label"
LABEL_NAMES = {0: "normal/benigno", 1: "outlier/ataque"}


def distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    rows = []
    for label, n in counts.items():
        label_int = int(label)
        rows.append(
            {
                "class": label_int,
                "name": LABEL_NAMES.get(label_int, "?"),
                "n": int(n),
                "pct": 100.0 * n / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    total = int(table["n"].sum())
    for _, row in table.iterrows():
        print(
            f"  classe {row['class']} ({row['name']:<15}): "
            f"{row['n']:>10,} amostras  ({row['pct']:6.2f}%)"
        )
    print(f"  {'TOTAL':<27}: {total:>10,} amostras")


def verdict(table: pd.DataFrame) -> None:
    if len(table) < 2:
        print("\n[!] Apenas uma classe presente")
        return

    counts_by_class = {int(r["class"]): int(r["n"]) for _, r in table.iterrows()}
    n_normal = counts_by_class.get(0, 0)
    n_outlier = counts_by_class.get(1, 0)

    majority = max(n_normal, n_outlier)
    minority = min(n_normal, n_outlier)
    ratio = majority / minority if minority else float("inf")
    contamination = 100.0 * n_outlier / (n_normal + n_outlier)

    print("-----------")
    print(f"  Razão de desbalanceamento (maj:min): {ratio:6.2f} : 1")
    print(f"  Contaminação (% de outliers)       : {contamination:6.2f}%")

    if ratio <= 1.5:
        verdict_text = "Balanceado (proporção ~1:1)."
    elif ratio <= 4:
        verdict_text = "Levemente desbalanceamento."
    else:
        verdict_text = "Fortemente desbalanceado."
    print(f"  Veredito                           : {verdict_text}")


def read_arff_header(path: Path) -> tuple[list[str], list[str]]:
    header_lines: list[str] = []
    attributes: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            header_lines.append(raw.rstrip("\n"))
            low = raw.strip().lower()
            if low.startswith("@attribute"):
                rest = raw.strip()[len("@attribute") :].strip()
                if rest.startswith("'"):
                    name = rest[1:].split("'", 1)[0]
                else:
                    name = rest.split()[0]
                attributes.append(name.strip())
            if low == "@data":
                break
    return header_lines, attributes


def apply_smote(
    df: pd.DataFrame, *, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN].astype(int)

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def write_arff(
    output: Path,
    header_lines: list[str],
    feature_cols: list[str],
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    feature_only_cols = [c for c in feature_cols if c != LABEL_COLUMN]
    X = X[feature_only_cols]

    data = np.column_stack([X.to_numpy(dtype=float), y.to_numpy(dtype=int)])

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header_lines) + "\n")
        np.savetxt(fh, data, fmt="%g", delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../data/Tuesday.arff"),
        help="Caminho do .arff a inspecionar (default: data/Tuesday.arff).",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Aplica SMOTE para balancear 50/50 e grava um novo .arff.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Saída do .arff balanceado (default: <dataset>_balanceado.arff).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente do SMOTE (reprodutibilidade). Default: 42.",
    )
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    df = load_arff(args.dataset)

    if LABEL_COLUMN not in df.columns:
        raise SystemExit(
            f"Coluna '{LABEL_COLUMN}' não encontrada. Colunas: {list(df.columns)[-5:]}"
        )

    table_raw = distribution(df[LABEL_COLUMN])
    print_table("Distribuição antes da limpeza", table_raw)

    df_clean, removed = clean_dataframe(df)
    print(f"\n[limpeza] {removed:,} linhas com inf/NaN removidas.")
    table_clean = distribution(df_clean[LABEL_COLUMN])
    print_table("Distribuição após limpeza", table_clean)

    verdict(table_clean)

    if not args.smote:
        return

    print("\n" + "=" * 60)
    print("Aplicando SMOTE...")
    X_resampled, y_resampled = apply_smote(df_clean, random_state=args.random_state)

    table_smote = distribution(y_resampled)
    print_table("Distribuição após SMOTE", table_smote)

    output = args.output or args.dataset.with_name(
        f"{args.dataset.stem}_balanceado.arff"
    )
    header_lines, attributes = read_arff_header(args.dataset)
    write_arff(output, header_lines, attributes, X_resampled, y_resampled)
    print(f"\n[ok] Dataset balanceado salvo em: {output}")
    print(
        f"     {len(y_resampled):,} linhas ({df_clean.shape[1] - 1} features + Label)."
    )


if __name__ == "__main__":
    main()
