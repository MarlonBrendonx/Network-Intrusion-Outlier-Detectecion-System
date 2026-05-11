"""
Domain Features para CIC-IDS2017
=================================

Sistema de registro de features de domínio que podem ser ativadas
seletivamente via CLI (--domain-features NAME1 NAME2) ou em conjunto
(--all-domain-features).

NOTA SOBRE NOMES DE COLUNAS:
Este módulo é específico para o schema do projeto NIOD, onde algumas
colunas do CIC-IDS2017 original foram renomeadas:
    - 'Total Fwd Packets'              → 'Soma Fwd Packets'
    - 'Total Length of Fwd Packets'    → 'Soma Length of Fwd Packets'
    - 'Total Backward Packets'         → não existe (usa 'Subflow Bwd Packets')
    - 'Total Length of Bwd Packets'    → não existe (usa 'Subflow Bwd Bytes')

API esperada pelo pipeline:
    - DOMAIN_FEATURES: dict[str, callable]  — registro de grupos
    - add_domain_features(df, features=None) -> df  — aplica os grupos
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1.0  # evita divisão por zero (1.0 mantém escala interpretável)

# ─────────────────────────────────────────────────────────────────────────────
# Aliases — nomes reais das colunas no schema do NIOD/CIC-IDS2017
# ─────────────────────────────────────────────────────────────────────────────
COL_FWD_PKTS = "Soma Fwd Packets"
COL_BWD_PKTS = "Subflow Bwd Packets"  # proxy para "Total Backward Packets"
COL_FWD_BYTES = "Soma Length of Fwd Packets"
COL_BWD_BYTES = "Subflow Bwd Bytes"  # proxy para "Total Length of Bwd Packets"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────
def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divisão segura: substitui inf/-inf/NaN por 0."""
    out = num / (den + EPS)
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    """Confere se todas as colunas existem; loga em DEBUG se faltar."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.debug("Colunas ausentes (pulando feature): %s", missing)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Grupo: Eng_Packet_Shape — forma/proporção dos pacotes
# Captura a "forma" da conversação independente do volume absoluto.
# ─────────────────────────────────────────────────────────────────────────────
def _add_eng_packet_shape(df: pd.DataFrame) -> list[str]:
    added = []

    # Razão Bwd/Fwd em bytes — assimetria do payload
    if _has_cols(df, [COL_BWD_BYTES, COL_FWD_BYTES]):
        df["bytes_ratio_bwd_fwd"] = _safe_div(df[COL_BWD_BYTES], df[COL_FWD_BYTES])
        added.append("bytes_ratio_bwd_fwd")

    # Razão Bwd/Fwd em pacotes — simetria da conversa
    if _has_cols(df, [COL_BWD_PKTS, COL_FWD_PKTS]):
        df["pkts_ratio_bwd_fwd"] = _safe_div(df[COL_BWD_PKTS], df[COL_FWD_PKTS])
        added.append("pkts_ratio_bwd_fwd")

    # Tamanho médio de pacote por direção
    if _has_cols(df, [COL_FWD_BYTES, COL_FWD_PKTS]):
        df["avg_pkt_size_fwd"] = _safe_div(df[COL_FWD_BYTES], df[COL_FWD_PKTS])
        added.append("avg_pkt_size_fwd")

    if _has_cols(df, [COL_BWD_BYTES, COL_BWD_PKTS]):
        df["avg_pkt_size_bwd"] = _safe_div(df[COL_BWD_BYTES], df[COL_BWD_PKTS])
        added.append("avg_pkt_size_bwd")

    # Razão entre tamanhos médios
    if {"avg_pkt_size_fwd", "avg_pkt_size_bwd"}.issubset(df.columns):
        df["pkt_size_ratio_bwd_fwd"] = _safe_div(
            df["avg_pkt_size_bwd"], df["avg_pkt_size_fwd"]
        )
        added.append("pkt_size_ratio_bwd_fwd")

    # Coeficiente de variação do tamanho de pacotes (homogeneidade)
    # Tráfego automatizado tende a ter pacotes muito uniformes.
    if _has_cols(df, ["Packet Length Std", "Packet Length Mean"]):
        df["pkt_len_cv"] = _safe_div(df["Packet Length Std"], df["Packet Length Mean"])
        added.append("pkt_len_cv")

    # Razão min/max — fingerprint de protocolo
    if _has_cols(df, ["Min Packet Length", "Max Packet Length"]):
        df["pkt_len_min_max_ratio"] = _safe_div(
            df["Min Packet Length"], df["Max Packet Length"]
        )
        added.append("pkt_len_min_max_ratio")

    return added


# ─────────────────────────────────────────────────────────────────────────────
# Grupo: Eng_Fwd_Header_Load — overhead de cabeçalho
# Razão entre bytes de cabeçalho e payload — handshakes/probes/scans
# têm muito cabeçalho e pouco payload.
# ─────────────────────────────────────────────────────────────────────────────
def _add_eng_fwd_header_load(df: pd.DataFrame) -> list[str]:
    added = []

    # Razão cabeçalho/payload em Fwd
    if _has_cols(df, ["Fwd Header Length", COL_FWD_BYTES]):
        df["fwd_header_to_payload_ratio"] = _safe_div(
            df["Fwd Header Length"], df[COL_FWD_BYTES]
        )
        added.append("fwd_header_to_payload_ratio")

    # Razão cabeçalho/payload em Bwd
    if _has_cols(df, ["Bwd Header Length", COL_BWD_BYTES]):
        df["bwd_header_to_payload_ratio"] = _safe_div(
            df["Bwd Header Length"], df[COL_BWD_BYTES]
        )
        added.append("bwd_header_to_payload_ratio")

    # Cabeçalho médio por pacote
    if _has_cols(df, ["Fwd Header Length", COL_FWD_PKTS]):
        df["fwd_header_per_pkt"] = _safe_div(df["Fwd Header Length"], df[COL_FWD_PKTS])
        added.append("fwd_header_per_pkt")

    if _has_cols(df, ["Bwd Header Length", COL_BWD_PKTS]):
        df["bwd_header_per_pkt"] = _safe_div(df["Bwd Header Length"], df[COL_BWD_PKTS])
        added.append("bwd_header_per_pkt")

    # Razão pacotes-com-payload / total-pacotes-fwd
    # Brute force tem muitos pacotes só de controle (handshakes, ACKs).
    if _has_cols(df, ["act_data_pkt_fwd", COL_FWD_PKTS]):
        df["fwd_data_pkt_ratio"] = _safe_div(df["act_data_pkt_fwd"], df[COL_FWD_PKTS])
        added.append("fwd_data_pkt_ratio")

    return added


# ─────────────────────────────────────────────────────────────────────────────
# Grupo: Eng_Temporal_Burstiness — regularidade temporal
# CV dos IATs detecta automatização: ataques tendem a ter IATs muito
# regulares (CV baixo) ou muito bursty (CV alto).
# ─────────────────────────────────────────────────────────────────────────────
def _add_eng_temporal_burstiness(df: pd.DataFrame) -> list[str]:
    added = []

    if _has_cols(df, ["Fwd IAT Std", "Fwd IAT Mean"]):
        df["fwd_iat_cv"] = _safe_div(df["Fwd IAT Std"], df["Fwd IAT Mean"])
        added.append("fwd_iat_cv")

    if _has_cols(df, ["Bwd IAT Std", "Bwd IAT Mean"]):
        df["bwd_iat_cv"] = _safe_div(df["Bwd IAT Std"], df["Bwd IAT Mean"])
        added.append("bwd_iat_cv")

    if _has_cols(df, ["Flow IAT Std", "Flow IAT Mean"]):
        df["flow_iat_cv"] = _safe_div(df["Flow IAT Std"], df["Flow IAT Mean"])
        added.append("flow_iat_cv")

    # Tempo médio por pacote (inverso da taxa global)
    if _has_cols(df, ["Flow Duration", COL_FWD_PKTS, COL_BWD_PKTS]):
        total_pkts = df[COL_FWD_PKTS] + df[COL_BWD_PKTS]
        df["duration_per_pkt"] = _safe_div(df["Flow Duration"], total_pkts)
        added.append("duration_per_pkt")

    # Razão Active/Idle — captura cadência de brute force
    # Brute force automatizado tem ciclos ativos curtos seguidos de idles
    # regulares; tráfego humano tem padrão mais irregular.
    if _has_cols(df, ["Active Mean", "Idle Mean"]):
        df["active_idle_ratio"] = _safe_div(df["Active Mean"], df["Idle Mean"])
        added.append("active_idle_ratio")

    # Variabilidade do período ativo (CV)
    if _has_cols(df, ["Active Std", "Active Mean"]):
        df["active_cv"] = _safe_div(df["Active Std"], df["Active Mean"])
        added.append("active_cv")

    return added


# ─────────────────────────────────────────────────────────────────────────────
# Grupo: Eng_Flag_Density — densidade de flags TCP
# Proporção de flags por pacote total. Brute force gera muitos handshakes
# (SYN/ACK altos), DDoS SYN flood gera SYN próximo de 100%.
# ─────────────────────────────────────────────────────────────────────────────
def _add_eng_flag_density(df: pd.DataFrame) -> list[str]:
    added = []

    if not _has_cols(df, [COL_FWD_PKTS, COL_BWD_PKTS]):
        return added

    total_pkts = df[COL_FWD_PKTS] + df[COL_BWD_PKTS]

    # Flags TCP — todos os nomes correspondentes ao schema do NIOD
    flag_cols = [
        "SYN Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "RST Flag Count",
        "URG Flag Count",
        "FIN Flag Count",
        "ECE Flag Count",
    ]

    for flag in flag_cols:
        if flag in df.columns:
            feat_name = f"{flag.lower().replace(' ', '_')}_density"
            df[feat_name] = _safe_div(df[flag], total_pkts)
            added.append(feat_name)

    # PSH/URG no Fwd separadamente (são contadores, não Count)
    if _has_cols(df, ["Fwd PSH Flags", COL_FWD_PKTS]):
        df["fwd_psh_density"] = _safe_div(df["Fwd PSH Flags"], df[COL_FWD_PKTS])
        added.append("fwd_psh_density")

    if _has_cols(df, ["Fwd URG Flags", COL_FWD_PKTS]):
        df["fwd_urg_density"] = _safe_div(df["Fwd URG Flags"], df[COL_FWD_PKTS])
        added.append("fwd_urg_density")

    return added


# ─────────────────────────────────────────────────────────────────────────────
# Grupo: Eng_Flow_Indicators — indicadores binários e razões compostas
# ─────────────────────────────────────────────────────────────────────────────
def _add_eng_flow_indicators(df: pd.DataFrame) -> list[str]:
    added = []

    # Fluxo unidirecional (sem resposta) — típico de SYN flood ou probe falho
    if COL_BWD_PKTS in df.columns:
        df["is_unidirectional"] = (df[COL_BWD_PKTS] == 0).astype(int)
        added.append("is_unidirectional")

    # Fluxo muito curto (< 100ms) — típico de probe/scan
    if "Flow Duration" in df.columns:
        # Flow Duration está em microssegundos no CIC-IDS2017
        df["is_short_flow"] = (df["Flow Duration"] < 100_000).astype(int)
        added.append("is_short_flow")

    # Razão entre janelas TCP iniciais — fingerprint cliente/servidor
    if _has_cols(df, ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]):
        df["init_win_ratio"] = _safe_div(
            df["Init_Win_bytes_forward"].clip(lower=0),
            df["Init_Win_bytes_backward"].clip(lower=0),
        )
        added.append("init_win_ratio")

    # Razão de subflows
    if _has_cols(df, ["Subflow Fwd Bytes", "Subflow Bwd Bytes"]):
        df["subflow_bytes_ratio"] = _safe_div(
            df["Subflow Bwd Bytes"], df["Subflow Fwd Bytes"]
        )
        added.append("subflow_bytes_ratio")

    # Indicador: sessão tem janela TCP forward zerada (cliente fechou rápido)
    if "Init_Win_bytes_forward" in df.columns:
        df["init_win_fwd_is_zero"] = (df["Init_Win_bytes_forward"] <= 0).astype(int)
        added.append("init_win_fwd_is_zero")

    # min_seg_size_forward é um fingerprint do TCP MSS — vale como feature
    # standalone, mas a razão com header capta o overhead relativo
    if _has_cols(df, ["min_seg_size_forward", "Fwd Header Length"]):
        df["fwd_min_seg_to_header"] = _safe_div(
            df["min_seg_size_forward"], df["Fwd Header Length"]
        )
        added.append("fwd_min_seg_to_header")

    return added


# ─────────────────────────────────────────────────────────────────────────────
# Grupo: Eng_Flow_Rates — taxa de pacotes e bytes por tempo
# Assinatura volumétrica de DoS: floods geram taxa de pacotes/bytes muito
# acima do tráfego normal. Slowloris gera taxa de bytes baixíssima por
# tempo longo (keepalive malicioso).
# ─────────────────────────────────────────────────────────────────────────────
def _add_eng_flow_rates(df: pd.DataFrame) -> list[str]:
    added = []

    if "Flow Duration" not in df.columns:
        return added

    duration = df["Flow Duration"]

    if _has_cols(df, [COL_FWD_PKTS, COL_BWD_PKTS]):
        total_pkts = df[COL_FWD_PKTS] + df[COL_BWD_PKTS]
        df["pkt_rate"] = _safe_div(total_pkts, duration)
        added.append("pkt_rate")

        df["fwd_pkt_rate"] = _safe_div(df[COL_FWD_PKTS], duration)
        added.append("fwd_pkt_rate")

        df["bwd_pkt_rate"] = _safe_div(df[COL_BWD_PKTS], duration)
        added.append("bwd_pkt_rate")

    if _has_cols(df, [COL_FWD_BYTES, COL_BWD_BYTES]):
        total_bytes = df[COL_FWD_BYTES] + df[COL_BWD_BYTES]
        df["byte_rate"] = _safe_div(total_bytes, duration)
        added.append("byte_rate")

        df["fwd_byte_rate"] = _safe_div(df[COL_FWD_BYTES], duration)
        added.append("fwd_byte_rate")

    return added


# ─────────────────────────────────────────────────────────────────────────────
# Registro público — usado pelo CLI (--all-domain-features) e por
# --domain-features NAME1 NAME2 ...
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_FEATURES: dict[str, Callable[[pd.DataFrame], list[str]]] = {
    "Eng_Packet_Shape": _add_eng_packet_shape,
    "Eng_Fwd_Header_Load": _add_eng_fwd_header_load,
    "Eng_Temporal_Burstiness": _add_eng_temporal_burstiness,
    "Eng_Flag_Density": _add_eng_flag_density,
    "Eng_Flow_Indicators": _add_eng_flow_indicators,
    "Eng_Flow_Rates": _add_eng_flow_rates,
}


# ─────────────────────────────────────────────────────────────────────────────
# Função pública — chamada pelo pipeline (data.py / generalization.py)
# ─────────────────────────────────────────────────────────────────────────────
def add_domain_features(
    df: pd.DataFrame,
    features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Aplica grupos de domain features ao DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas originais do CIC-IDS2017 (schema NIOD).
    features : iterable of str | None
        Lista de nomes de grupos a aplicar. Se None ou vazio, retorna df
        inalterado. Nomes desconhecidos geram warning e são ignorados.

    Returns
    -------
    pd.DataFrame
        Cópia do DataFrame com as features adicionadas.
    """
    if not features:
        return df

    features = list(features)

    logger.info("=" * 70)
    logger.info("[domain_features] Aplicando grupos: %s", features)
    logger.info("[domain_features] Shape de entrada: %s", df.shape)

    df = df.copy()
    n_before = df.shape[1]

    for name in features:
        if name not in DOMAIN_FEATURES:
            logger.warning(
                "[domain_features] Grupo desconhecido '%s' — ignorando. "
                "Disponíveis: %s",
                name,
                list(DOMAIN_FEATURES.keys()),
            )
            continue

        added = DOMAIN_FEATURES[name](df)
        logger.info(
            "[domain_features] %s: %d features adicionadas (%s)",
            name,
            len(added),
            added,
        )

    logger.info(
        "[domain_features] Total: %d novas features. Shape de saída: %s",
        df.shape[1] - n_before,
        df.shape,
    )
    logger.info("=" * 70)

    return df
