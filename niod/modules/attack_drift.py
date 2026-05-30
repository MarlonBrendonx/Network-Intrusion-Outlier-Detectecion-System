"""
Análise de drift nos padrões de ataque entre dois datasets.

Compara as distribuições das amostras de ataque (Label=1) entre
dois dias usando o teste KS feature-a-feature, revelando se os
ataques têm assinaturas estatisticamente distintas.

Uso direto:
    python -m niod.modules.attack_drift
    python -m niod.modules.attack_drift --train data/Friday_balanceado.arff --target data/Tuesday.arff
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


@dataclass
class AttackDriftResult:
    """Resultado da comparação de padrões de ataque entre dois datasets."""

    train_name: str
    target_name: str
    train_attack_count: int
    target_attack_count: int
    statistics: dict[str, float]
    high_drift: list[str]
    medium_drift: list[str]
    high_drift_ratio: float
    medium_drift_ratio: float


def _load_attacks(path: Path, label_column: str = "Label") -> pd.DataFrame:
    """Carrega apenas amostras de ataque (label=1) de um arquivo .arff."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df[df[label_column] == 1].drop(columns=[label_column])


def analyze_attack_drift(
    train_path: Path,
    target_path: Path,
    *,
    high_threshold: float = 0.3,
    medium_threshold: float = 0.1,
    label_column: str = "Label",
) -> AttackDriftResult:
    """
    Compara distribuições de ataque entre dois datasets via teste KS.

    Args:
        train_path: Dataset de treino (referência).
        target_path: Dataset alvo (generalização).
        high_threshold: KS acima deste valor indica drift severo.
        medium_threshold: KS acima deste valor indica drift moderado.
        label_column: Nome da coluna de label.

    Returns:
        AttackDriftResult com estatísticas por feature e resumo de drift.
    """
    logger.info("Carregando ataques de %s...", train_path.stem)
    train_attacks = _load_attacks(train_path, label_column)

    logger.info("Carregando ataques de %s...", target_path.stem)
    target_attacks = _load_attacks(target_path, label_column)

    shared_features = [c for c in train_attacks.columns if c in target_attacks.columns]

    statistics: dict[str, float] = {}
    for col in shared_features:
        stat, _ = ks_2samp(train_attacks[col].values, target_attacks[col].values)
        statistics[col] = round(float(stat), 4)

    high_drift = [f for f, ks in statistics.items() if ks > high_threshold]
    medium_drift = [f for f, ks in statistics.items() if medium_threshold < ks <= high_threshold]
    n = len(statistics)

    return AttackDriftResult(
        train_name=train_path.stem,
        target_name=target_path.stem,
        train_attack_count=len(train_attacks),
        target_attack_count=len(target_attacks),
        statistics=statistics,
        high_drift=high_drift,
        medium_drift=medium_drift,
        high_drift_ratio=len(high_drift) / n if n else 0.0,
        medium_drift_ratio=len(medium_drift) / n if n else 0.0,
    )


def print_report(result: AttackDriftResult, top_n: int = 20) -> None:
    """Imprime relatório legível do drift de ataque."""
    print("=" * 70)
    print("ANÁLISE DE DRIFT NOS PADRÕES DE ATAQUE")
    print("=" * 70)
    print(f"Treino  ({result.train_name}):  {result.train_attack_count:,} ataques")
    print(f"Alvo    ({result.target_name}): {result.target_attack_count:,} ataques")
    print()

    sorted_stats = sorted(result.statistics.items(), key=lambda x: x[1], reverse=True)

    print(f"Top {top_n} features mais divergentes entre os ataques:")
    print("-" * 60)
    for feat, ks in sorted_stats[:top_n]:
        if ks > 0.3:
            tag = " ← SEVERO"
        elif ks > 0.1:
            tag = " ← moderado"
        else:
            tag = ""
        print(f"  {feat:<45} KS={ks:.3f}{tag}")

    print()
    print("Resumo:")
    n = len(result.statistics)
    print(f"  Drift severo   (KS > 0.3): {len(result.high_drift):>3}/{n} ({result.high_drift_ratio * 100:.1f}%)")
    print(f"  Drift moderado (KS > 0.1): {len(result.medium_drift):>3}/{n} ({result.medium_drift_ratio * 100:.1f}%)")
    print("=" * 70)

    if result.high_drift_ratio > 0.3:
        print("CONCLUSÃO: Ataques com padrões muito distintos entre os dias.")
        print("           Modelos não-supervisionados têm limitação estrutural aqui.")
        print("           Recomendado: classificação supervisionada.")
    elif result.high_drift_ratio > 0.1:
        print("CONCLUSÃO: Drift moderado. Enriquecimento do treino pode ajudar.")
    else:
        print("CONCLUSÃO: Ataques similares entre os dias. Modelo deve generalizar bem.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Análise de drift nos padrões de ataque entre dois datasets."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/Friday_balanceado.arff"),
        help="Dataset de treino (referência)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/Tuesday.arff"),
        help="Dataset alvo (generalização)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.3,
        help="KS acima deste valor = drift severo",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.1,
        help="KS acima deste valor = drift moderado",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Quantas features exibir no relatório",
    )
    args = parser.parse_args()

    result = analyze_attack_drift(
        train_path=args.train,
        target_path=args.target,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
    )
    print_report(result, top_n=args.top_n)


if __name__ == "__main__":
    main()
