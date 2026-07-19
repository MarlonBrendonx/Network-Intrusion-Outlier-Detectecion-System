from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from niod.utils.data import clean_dataframe, load_arff, prepare_splits

CUTS = [
    (9, "Elbow (~80%)", "#d62728"),
    (12, "Intermediate (~88%)", "#ff7f0e"),
    (16, "95% threshold", "#2ca02c"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("data/Friday.arff"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()

    df = load_arff(args.train_dataset)
    df, _ = clean_dataframe(df)
    split = prepare_splits(
        df,
        label_column="Label",
        novelty=True,
        algorithm="lof",
        contamination=None,
        train_size=0.6,
        random_state=42,
        apply_filters=True,
        pca_reduce=None,
        domain_features=None,
    )
    X = np.asarray(split.X_train)
    n_feat = X.shape[1]

    pca = PCA(n_components=None, random_state=42).fit(X)
    evr = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(evr)
    comps = np.arange(1, n_feat + 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.bar(comps, evr, color="#4c72b0", alpha=0.7, width=0.8)
    ax1.plot(comps, evr, color="#1f3b6f", marker="o", ms=3, lw=1)
    ax1.set_title(f"Scree plot — {n_feat} filtered features (train)")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.axvline(9, color="#d62728", ls="--", lw=1.2, alpha=0.8)
    ax1.annotate(
        "elbow ≈ 9",
        xy=(9, evr[8]),
        xytext=(14, evr[8] + 4),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        color="#d62728",
        fontsize=10,
    )
    ax1.set_xlim(0, n_feat + 1)
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    out1 = args.output_dir / "pca_scree.png"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"Figure saved to: {out1}")

    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    ax2.plot(comps, cum, color="#1f3b6f", marker="o", ms=3, lw=1.5)
    ax2.set_title(f"Cumulative variance — {n_feat} filtered features (train)")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_xlim(0, n_feat + 1)
    ax2.set_ylim(0, 102)
    ax2.grid(alpha=0.3)

    for n, label, color in CUTS:
        y = cum[n - 1]
        ax2.axvline(n, color=color, ls="--", lw=1.2, alpha=0.8)
        ax2.scatter([n], [y], color=color, zorder=5, s=40)
        ax2.annotate(
            f"N={n}\n{y:.1f}%",
            xy=(n, y),
            xytext=(n + 1.5, y - 14),
            color=color,
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color),
        )
    fig2.tight_layout()
    out2 = args.output_dir / "pca_cumulative.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"Figure saved to: {out2}")

    print(f"\nPCA input features: {n_feat}")
    print("\nCumulative variance at the evaluated cuts:")
    for n, label, _ in CUTS:
        print(f"  N={n:2d} ({label:22s}): {cum[n-1]:5.1f}%")
    print("\nComponents for reference thresholds:")
    for thr in (80, 90, 95, 99):
        n = int(np.searchsorted(cum, thr) + 1)
        print(f"  {thr}% -> {n} components")


if __name__ == "__main__":
    main()
