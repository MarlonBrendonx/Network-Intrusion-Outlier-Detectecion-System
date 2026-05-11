"""
Módulo de visualização para análise de drift e generalização.

Gera gráficos UMAP 3D comparando datasets de treino vs. alvo,
facilitando a identificação visual de concept drift.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from niod.config.settings import ExperimentConfig

logger = logging.getLogger(__name__)


def generate_umap_3d(
    transformer,
    columns_ref: pd.Index,
    train_means: pd.Series,
    train_dataset_path: Path,
    target_dataset_path: Path,
    config: ExperimentConfig,
    output_path: Path = Path("umap_3d_generalization_analysis.png"),
    *,
    interactive: bool = False,
) -> Path:
    """
    Gera visualização UMAP 3D comparando domínio de treino vs. alvo.

    O UMAP é fitado APENAS nos dados normais do treino, preservando
    a perspectiva do modelo de novelty detection.

    Args:
        transformer: Pipeline de transformação (scaler) treinado.
        columns_ref: Colunas de features de referência.
        train_means: Médias do treino para imputação consistente.
        train_dataset_path: Caminho do dataset de treino.
        target_dataset_path: Caminho do dataset alvo.
        config: Configurações do experimento.
        output_path: Caminho para salvar o gráfico.

    Returns:
        Caminho do arquivo salvo.
    """
    from niod.utils.data import prepare_data_for_visualization

    train_stem = train_dataset_path.stem
    target_stem = target_dataset_path.stem

    logger.info(
        "Gerando UMAP 3D: %s (treino) vs %s (alvo)",
        train_stem,
        target_stem,
    )

    # Carregar e preparar dados
    X_train_norm, X_train_att = prepare_data_for_visualization(
        train_dataset_path,
        columns_ref,
        train_means,
        f"{train_stem} (Source)",
        sample_size=config.umap_sample_size,
    )

    X_target_norm, X_target_att = prepare_data_for_visualization(
        target_dataset_path,
        columns_ref,
        train_means,
        f"{target_stem} (Target)",
        sample_size=config.umap_sample_size,
    )

    # Aplicar transformação do pipeline treinado
    logger.info("Aplicando transformação do pipeline...")
    data_train_norm = transformer.transform(X_train_norm)
    data_train_att = transformer.transform(X_train_att)
    data_target_norm = transformer.transform(X_target_norm)
    data_target_att = transformer.transform(X_target_att)

    # UMAP 3D — fit apenas nos normais do treino
    logger.info("Treinando UMAP 3D (fit em %s normal)...", train_stem)
    umap_vis = umap.UMAP(
        n_components=config.umap_n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_state,
        n_jobs=-1,
    )
    umap_vis.fit(data_train_norm)

    # Projetar todos os conjuntos
    logger.info("Projetando dados em 3D...")
    p_train_norm = umap_vis.transform(data_train_norm)
    p_train_att = (
        umap_vis.transform(data_train_att)
        if len(data_train_att) > 0
        else np.empty((0, config.umap_n_components))
    )
    p_target_norm = umap_vis.transform(data_target_norm)
    p_target_att = (
        umap_vis.transform(data_target_att)
        if len(data_target_att) > 0
        else np.empty((0, config.umap_n_components))
    )

    if interactive:
        output_path = _plot_umap_interactive(
            p_train_norm, p_train_att,
            p_target_norm, p_target_att,
            train_stem, target_stem,
            config.algorithm.value,
            output_path,
        )
    else:
        output_path = _plot_umap_static(
            p_train_norm, p_train_att,
            p_target_norm, p_target_att,
            train_stem, target_stem,
            config.algorithm.value,
            output_path,
        )

    return output_path


def _plot_umap_static(
    p_train_norm, p_train_att,
    p_target_norm, p_target_att,
    train_stem: str, target_stem: str,
    algorithm: str,
    output_path: Path,
) -> Path:
    """Salva PNG estático com matplotlib."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        p_train_norm[:, 0], p_train_norm[:, 1], p_train_norm[:, 2],
        c="lightgrey", label=f"{train_stem}: Normal (Baseline)", alpha=0.2, s=10,
    )
    if len(p_train_att) > 0:
        ax.scatter(
            p_train_att[:, 0], p_train_att[:, 1], p_train_att[:, 2],
            c="black", marker="x", label=f"{train_stem}: Ataque", alpha=0.5, s=25,
        )
    ax.scatter(
        p_target_norm[:, 0], p_target_norm[:, 1], p_target_norm[:, 2],
        c="blue", label=f"{target_stem}: Normal", alpha=0.3, s=10,
    )
    if len(p_target_att) > 0:
        ax.scatter(
            p_target_att[:, 0], p_target_att[:, 1], p_target_att[:, 2],
            c="red", marker="^", label=f"{target_stem}: Ataque", alpha=0.8, s=40,
        )

    ax.set_title(f"Análise de Drift: UMAP 3D ({algorithm})", fontsize=16)
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    ax.set_zlabel("UMAP Dim 3")
    ax.legend(loc="best", fontsize=9)

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico UMAP 3D salvo: %s", output_path)
    return output_path


def _plot_umap_interactive(
    p_train_norm, p_train_att,
    p_target_norm, p_target_att,
    train_stem: str, target_stem: str,
    algorithm: str,
    output_path: Path,
) -> Path:
    """Salva HTML interativo com Plotly (rotacionável no browser)."""
    import plotly.graph_objects as go

    traces = [
        go.Scatter3d(
            x=p_train_norm[:, 0], y=p_train_norm[:, 1], z=p_train_norm[:, 2],
            mode="markers",
            name=f"{train_stem}: Normal",
            marker=dict(size=2, color="lightgrey", opacity=0.3),
        ),
        go.Scatter3d(
            x=p_target_norm[:, 0], y=p_target_norm[:, 1], z=p_target_norm[:, 2],
            mode="markers",
            name=f"{target_stem}: Normal",
            marker=dict(size=2, color="steelblue", opacity=0.4),
        ),
    ]

    if len(p_train_att) > 0:
        traces.append(go.Scatter3d(
            x=p_train_att[:, 0], y=p_train_att[:, 1], z=p_train_att[:, 2],
            mode="markers",
            name=f"{train_stem}: Ataque",
            marker=dict(size=4, color="black", symbol="cross", opacity=0.7),
        ))

    if len(p_target_att) > 0:
        traces.append(go.Scatter3d(
            x=p_target_att[:, 0], y=p_target_att[:, 1], z=p_target_att[:, 2],
            mode="markers",
            name=f"{target_stem}: Ataque",
            marker=dict(size=5, color="red", symbol="diamond", opacity=0.9),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Análise de Drift: UMAP 3D ({algorithm})",
        scene=dict(
            xaxis_title="UMAP Dim 1",
            yaxis_title="UMAP Dim 2",
            zaxis_title="UMAP Dim 3",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    output_path = Path(str(output_path).replace(".png", ".html"))
    fig.write_html(str(output_path))
    logger.info("Gráfico UMAP 3D interativo salvo: %s", output_path)
    return output_path
