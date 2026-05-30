"""
Módulo de visualização PCA.

Gera projeção 2D ou 3D dos dados usando PCA, separando visualmente
amostras Normal vs Outlier. Útil para diagnóstico da qualidade do
espaço de features (especialmente após aplicação de filtros estatísticos).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def generate_pca_plot(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_components: Literal[2, 3] = 2,
    output_path: Path = Path("pca_class_separation.png"),
    title: str = "PCA — Separação Normal vs Outlier",
    sample_size: Optional[int] = 10000,
    random_state: int = 42,
    standardize: bool = True,
) -> Path:
    """
    Projeta `X` em 2D ou 3D via PCA e plota Normal vs Outlier.

    Args:
        X: Matriz de features (n_samples, n_features). Espera-se que já
           esteja imputada e filtrada (saída de `prepare_splits`).
        y: Labels binárias (0 = Normal, 1 = Outlier) ou no formato sklearn
           (1 = Normal, -1 = Outlier). A função detecta automaticamente.
        n_components: 2 para gráfico 2D, 3 para 3D.
        output_path: Onde salvar o gráfico (PNG).
        title: Título do gráfico.
        sample_size: Se fornecido, amostra esse número de pontos por classe
                     (para gráficos legíveis em datasets grandes).
        random_state: Semente para amostragem reprodutível.
        standardize: Se True, aplica StandardScaler antes do PCA. PCA
                     é sensível à escala — manter ligado é o default seguro.

    Returns:
        Caminho do arquivo salvo.
    """
    if n_components not in (2, 3):
        raise ValueError(f"n_components deve ser 2 ou 3, recebido: {n_components}")

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Normalizar labels para {0, 1}: 0 = Normal, 1 = Outlier
    if set(np.unique(y)).issubset({-1, 1}):
        # Formato sklearn: 1 = Normal, -1 = Outlier → 0/1
        y_bin = np.where(y == -1, 1, 0)
    else:
        y_bin = (y > 0).astype(int)

    n_normal = int((y_bin == 0).sum())
    n_outlier = int((y_bin == 1).sum())
    logger.info(
        "PCA — entrada: %d amostras, %d features (Normal=%d, Outlier=%d)",
        X.shape[0],
        X.shape[1],
        n_normal,
        n_outlier,
    )

    # Subamostragem por classe (gráfico legível)
    if sample_size is not None:
        rng = np.random.default_rng(random_state)
        idx_normal = np.where(y_bin == 0)[0]
        idx_outlier = np.where(y_bin == 1)[0]

        if len(idx_normal) > sample_size:
            idx_normal = rng.choice(idx_normal, size=sample_size, replace=False)
        if len(idx_outlier) > sample_size:
            idx_outlier = rng.choice(idx_outlier, size=sample_size, replace=False)

        keep = np.concatenate([idx_normal, idx_outlier])
        X = X[keep]
        y_bin = y_bin[keep]
        logger.info(
            "PCA — após subamostragem: %d Normal, %d Outlier",
            (y_bin == 0).sum(),
            (y_bin == 1).sum(),
        )

    # Padronização (PCA é sensível à escala)
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Ajuste do PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    total_var = explained.sum() * 100
    logger.info(
        "PCA — variância explicada por componente: %s | total: %.2f%%",
        [f"{v:.2%}" for v in explained],
        total_var,
    )

    # ---- Plot ------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))

    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(
            X_pca[y_bin == 0, 0],
            X_pca[y_bin == 0, 1],
            c="#1f77b4",
            s=8,
            alpha=0.5,
            label=f"Normal (n={n_normal:,})",
        )
        ax.scatter(
            X_pca[y_bin == 1, 0],
            X_pca[y_bin == 1, 1],
            c="#d62728",
            s=8,
            alpha=0.6,
            label=f"Outlier (n={n_outlier:,})",
        )
        ax.set_xlabel(f"PC1 ({explained[0]:.2%} variância)")
        ax.set_ylabel(f"PC2 ({explained[1]:.2%} variância)")

    else:  # 3D
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            X_pca[y_bin == 0, 0],
            X_pca[y_bin == 0, 1],
            X_pca[y_bin == 0, 2],
            c="#1f77b4",
            s=8,
            alpha=0.5,
            label=f"Normal (n={n_normal:,})",
        )
        ax.scatter(
            X_pca[y_bin == 1, 0],
            X_pca[y_bin == 1, 1],
            X_pca[y_bin == 1, 2],
            c="#d62728",
            s=8,
            alpha=0.6,
            label=f"Outlier (n={n_outlier:,})",
        )
        ax.set_xlabel(f"PC1 ({explained[0]:.2%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.2%})")
        ax.set_zlabel(f"PC3 ({explained[2]:.2%})")

    ax.set_title(f"{title}\n(variância total explicada: {total_var:.1f}%)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("Gráfico PCA salvo em: %s", output_path)
    return output_path


def generate_pca_cross_domain(
    X_train_normal: np.ndarray,
    X_train_attack: np.ndarray,
    X_gen_normal: np.ndarray,
    X_gen_attack: np.ndarray,
    *,
    train_label: str = "Friday",
    gen_label: str = "Tuesday",
    output_path: Path = Path("pca_cross_domain.png"),
    sample_size: Optional[int] = 5000,
    random_state: int = 42,
    n_components: Literal[2, 3] = 2,
    interactive: bool = False,
) -> Path:
    """
    Plota PCA 2D ou 3D combinando duas distribuições (treino vs generalização)
    com 4 categorias para evidenciar o deslocamento de domínio.

    O PCA é ajustado em TODOS os pontos juntos (treino + generalização),
    para que a projeção seja comparável entre os dois dias na mesma escala.

    Args:
        X_train_normal: Amostras normais do dataset de treino.
        X_train_attack: Amostras de ataque do dataset de treino.
        X_gen_normal: Amostras normais do dataset de generalização.
        X_gen_attack: Amostras de ataque do dataset de generalização.
        train_label: Nome do dataset de treino (ex: "Friday").
        gen_label: Nome do dataset de generalização (ex: "Tuesday").
        output_path: Caminho do PNG (ou HTML, se interactive) de saída.
        sample_size: Pontos por categoria (subamostragem para legibilidade).
        random_state: Semente reprodutível.
        n_components: 2 (default) ou 3 — dimensionalidade da projeção.
        interactive: Se True, gera HTML via Plotly (rotacionável); se False,
            PNG estático via Matplotlib. Útil sobretudo no 3D, onde o
            ângulo de câmera estático pode esconder estrutura.

    Returns:
        Caminho do arquivo salvo.
    """
    rng = np.random.default_rng(random_state)

    def _subsample(X: np.ndarray, n: Optional[int]) -> np.ndarray:
        X = np.asarray(X)
        if n is None or len(X) <= n:
            return X
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx]

    X_tn = _subsample(X_train_normal, sample_size)
    X_ta = _subsample(X_train_attack, sample_size)
    X_gn = _subsample(X_gen_normal, sample_size)
    X_ga = _subsample(X_gen_attack, sample_size)

    logger.info(
        "PCA cross-domain — %s Normal: %d | %s Ataque: %d | %s Normal: %d | %s Ataque: %d",
        train_label,
        len(X_tn),
        train_label,
        len(X_ta),
        gen_label,
        len(X_gn),
        gen_label,
        len(X_ga),
    )

    # Empilha tudo e marca origem (0..3) para split posterior
    X_all = np.vstack([X_tn, X_ta, X_gn, X_ga])
    sizes = [len(X_tn), len(X_ta), len(X_gn), len(X_ga)]
    cuts = np.cumsum([0] + sizes)

    if n_components not in (2, 3):
        raise ValueError(f"n_components deve ser 2 ou 3, recebido: {n_components}")

    # Padronização e PCA ajustados no conjunto unificado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    logger.info(
        "PCA cross-domain — variância explicada: %s | total: %.2f%%",
        " | ".join(f"PC{i+1}={v * 100:.2f}%" for i, v in enumerate(explained)),
        explained.sum() * 100,
    )

    pts_tn = X_pca[cuts[0] : cuts[1]]
    pts_ta = X_pca[cuts[1] : cuts[2]]
    pts_gn = X_pca[cuts[2] : cuts[3]]
    pts_ga = X_pca[cuts[3] : cuts[4]]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if interactive:
        return _plot_pca_cross_interactive(
            pts_tn, pts_ta, pts_gn, pts_ga,
            explained=explained,
            train_label=train_label,
            gen_label=gen_label,
            n_components=n_components,
            output_path=output_path,
        )
    return _plot_pca_cross_static(
        pts_tn, pts_ta, pts_gn, pts_ga,
        explained=explained,
        train_label=train_label,
        gen_label=gen_label,
        n_components=n_components,
        output_path=output_path,
    )


def _plot_pca_cross_static(
    pts_tn: np.ndarray,
    pts_ta: np.ndarray,
    pts_gn: np.ndarray,
    pts_ga: np.ndarray,
    *,
    explained: np.ndarray,
    train_label: str,
    gen_label: str,
    n_components: int,
    output_path: Path,
) -> Path:
    """Salva PNG estático com matplotlib (2D ou 3D)."""
    fig = plt.figure(figsize=(11, 8) if n_components == 3 else (11, 7))

    if n_components == 2:
        ax = fig.add_subplot(111)
        # Camadas: normais primeiro (mais leves), ataques por cima (mais visíveis)
        ax.scatter(
            pts_tn[:, 0], pts_tn[:, 1],
            c="#bdbdbd", s=10, alpha=0.45, marker="o",
            label=f"{train_label}: Normal (Treino)",
        )
        ax.scatter(
            pts_gn[:, 0], pts_gn[:, 1],
            c="#1f77b4", s=10, alpha=0.45, marker="o",
            label=f"{gen_label}: Normal",
        )
        ax.scatter(
            pts_ta[:, 0], pts_ta[:, 1],
            c="#000000", s=18, alpha=0.7, marker="x",
            label=f"{train_label}: Ataques",
        )
        ax.scatter(
            pts_ga[:, 0], pts_ga[:, 1],
            c="#d62728", s=22, alpha=0.75, marker="^",
            label=f"{gen_label}: Ataques",
        )
        ax.set_xlabel(f"PC1 ({explained[0]:.2%} variância)")
        ax.set_ylabel(f"PC2 ({explained[1]:.2%} variância)")
        ax.grid(True, alpha=0.3)
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            pts_tn[:, 0], pts_tn[:, 1], pts_tn[:, 2],
            c="#bdbdbd", s=10, alpha=0.40, marker="o",
            label=f"{train_label}: Normal (Treino)",
        )
        ax.scatter(
            pts_gn[:, 0], pts_gn[:, 1], pts_gn[:, 2],
            c="#1f77b4", s=10, alpha=0.40, marker="o",
            label=f"{gen_label}: Normal",
        )
        ax.scatter(
            pts_ta[:, 0], pts_ta[:, 1], pts_ta[:, 2],
            c="#000000", s=20, alpha=0.75, marker="x",
            label=f"{train_label}: Ataques",
        )
        ax.scatter(
            pts_ga[:, 0], pts_ga[:, 1], pts_ga[:, 2],
            c="#d62728", s=24, alpha=0.80, marker="^",
            label=f"{gen_label}: Ataques",
        )
        ax.set_xlabel(f"PC1 ({explained[0]:.2%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.2%})")
        ax.set_zlabel(f"PC3 ({explained[2]:.2%})")

    ax.set_title(
        f"Visualização do Espaço de Características ({n_components}D): {train_label} vs {gen_label}\n"
        f"(variância total explicada: {explained.sum():.1%})"
    )
    ax.legend(loc="best", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("Gráfico PCA cross-domain salvo em: %s", output_path)
    return output_path


def _plot_pca_cross_interactive(
    pts_tn: np.ndarray,
    pts_ta: np.ndarray,
    pts_gn: np.ndarray,
    pts_ga: np.ndarray,
    *,
    explained: np.ndarray,
    train_label: str,
    gen_label: str,
    n_components: int,
    output_path: Path,
) -> Path:
    """Salva HTML interativo com Plotly (rotacionável/zoomable no browser)."""
    import plotly.graph_objects as go

    is_3d = n_components == 3
    ScatterCls = go.Scatter3d if is_3d else go.Scatter

    def _coords(pts: np.ndarray) -> dict:
        if is_3d:
            return dict(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2])
        return dict(x=pts[:, 0], y=pts[:, 1])

    traces = [
        ScatterCls(
            **_coords(pts_tn),
            mode="markers",
            name=f"{train_label}: Normal (Treino)",
            marker=dict(size=3 if is_3d else 5, color="lightgrey", opacity=0.40),
        ),
        ScatterCls(
            **_coords(pts_gn),
            mode="markers",
            name=f"{gen_label}: Normal",
            marker=dict(size=3 if is_3d else 5, color="steelblue", opacity=0.40),
        ),
    ]
    if len(pts_ta) > 0:
        traces.append(ScatterCls(
            **_coords(pts_ta),
            mode="markers",
            name=f"{train_label}: Ataques",
            marker=dict(size=4 if is_3d else 7, color="black", symbol="cross", opacity=0.75),
        ))
    if len(pts_ga) > 0:
        traces.append(ScatterCls(
            **_coords(pts_ga),
            mode="markers",
            name=f"{gen_label}: Ataques",
            marker=dict(size=5 if is_3d else 8, color="red", symbol="diamond", opacity=0.85),
        ))

    fig = go.Figure(data=traces)
    title = (
        f"PCA Cross-Domain ({n_components}D): {train_label} vs {gen_label} "
        f"— variância total: {explained.sum():.1%}"
    )
    if is_3d:
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f"PC1 ({explained[0]:.2%})",
                yaxis_title=f"PC2 ({explained[1]:.2%})",
                zaxis_title=f"PC3 ({explained[2]:.2%})",
            ),
            legend=dict(itemsizing="constant"),
            margin=dict(l=0, r=0, b=0, t=50),
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title=f"PC1 ({explained[0]:.2%})",
            yaxis_title=f"PC2 ({explained[1]:.2%})",
            legend=dict(itemsizing="constant"),
            margin=dict(l=0, r=0, b=0, t=50),
        )

    output_path = Path(str(output_path).replace(".png", ".html"))
    if output_path.suffix != ".html":
        output_path = output_path.with_suffix(".html")
    fig.write_html(str(output_path))
    logger.info("Gráfico PCA cross-domain interativo salvo em: %s", output_path)
    return output_path
