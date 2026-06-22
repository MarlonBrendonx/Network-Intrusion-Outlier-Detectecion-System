from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1.0

COL_FWD_PKTS = "Soma Fwd Packets"
COL_BWD_PKTS = "Subflow Bwd Packets"
COL_FWD_BYTES = "Soma Length of Fwd Packets"
COL_BWD_BYTES = "Subflow Bwd Bytes"


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / (den + EPS)
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.debug("Colunas ausentes (pulando feature): %s", missing)
        return False
    return True


def _add_eng_packet_shape(df: pd.DataFrame) -> list[str]:
    added = []

    if _has_cols(df, [COL_BWD_BYTES, COL_FWD_BYTES]):
        df["bytes_ratio_bwd_fwd"] = _safe_div(df[COL_BWD_BYTES], df[COL_FWD_BYTES])
        added.append("bytes_ratio_bwd_fwd")

    if _has_cols(df, [COL_BWD_PKTS, COL_FWD_PKTS]):
        df["pkts_ratio_bwd_fwd"] = _safe_div(df[COL_BWD_PKTS], df[COL_FWD_PKTS])
        added.append("pkts_ratio_bwd_fwd")

    if _has_cols(df, [COL_FWD_BYTES, COL_FWD_PKTS]):
        df["avg_pkt_size_fwd"] = _safe_div(df[COL_FWD_BYTES], df[COL_FWD_PKTS])
        added.append("avg_pkt_size_fwd")

    if _has_cols(df, [COL_BWD_BYTES, COL_BWD_PKTS]):
        df["avg_pkt_size_bwd"] = _safe_div(df[COL_BWD_BYTES], df[COL_BWD_PKTS])
        added.append("avg_pkt_size_bwd")

    if {"avg_pkt_size_fwd", "avg_pkt_size_bwd"}.issubset(df.columns):
        df["pkt_size_ratio_bwd_fwd"] = _safe_div(
            df["avg_pkt_size_bwd"], df["avg_pkt_size_fwd"]
        )
        added.append("pkt_size_ratio_bwd_fwd")

    if _has_cols(df, ["Packet Length Std", "Packet Length Mean"]):
        df["pkt_len_cv"] = _safe_div(df["Packet Length Std"], df["Packet Length Mean"])
        added.append("pkt_len_cv")

    if _has_cols(df, ["Min Packet Length", "Max Packet Length"]):
        df["pkt_len_min_max_ratio"] = _safe_div(
            df["Min Packet Length"], df["Max Packet Length"]
        )
        added.append("pkt_len_min_max_ratio")

    return added


def _add_eng_fwd_header_load(df: pd.DataFrame) -> list[str]:
    added = []

    if _has_cols(df, ["Fwd Header Length", COL_FWD_BYTES]):
        df["fwd_header_to_payload_ratio"] = _safe_div(
            df["Fwd Header Length"], df[COL_FWD_BYTES]
        )
        added.append("fwd_header_to_payload_ratio")

    if _has_cols(df, ["Bwd Header Length", COL_BWD_BYTES]):
        df["bwd_header_to_payload_ratio"] = _safe_div(
            df["Bwd Header Length"], df[COL_BWD_BYTES]
        )
        added.append("bwd_header_to_payload_ratio")

    if _has_cols(df, ["Fwd Header Length", COL_FWD_PKTS]):
        df["fwd_header_per_pkt"] = _safe_div(df["Fwd Header Length"], df[COL_FWD_PKTS])
        added.append("fwd_header_per_pkt")

    if _has_cols(df, ["Bwd Header Length", COL_BWD_PKTS]):
        df["bwd_header_per_pkt"] = _safe_div(df["Bwd Header Length"], df[COL_BWD_PKTS])
        added.append("bwd_header_per_pkt")

    if _has_cols(df, ["act_data_pkt_fwd", COL_FWD_PKTS]):
        df["fwd_data_pkt_ratio"] = _safe_div(df["act_data_pkt_fwd"], df[COL_FWD_PKTS])
        added.append("fwd_data_pkt_ratio")

    return added


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

    if _has_cols(df, ["Flow Duration", COL_FWD_PKTS, COL_BWD_PKTS]):
        total_pkts = df[COL_FWD_PKTS] + df[COL_BWD_PKTS]
        df["duration_per_pkt"] = _safe_div(df["Flow Duration"], total_pkts)
        added.append("duration_per_pkt")

    if _has_cols(df, ["Active Mean", "Idle Mean"]):
        df["active_idle_ratio"] = _safe_div(df["Active Mean"], df["Idle Mean"])
        added.append("active_idle_ratio")

    if _has_cols(df, ["Active Std", "Active Mean"]):
        df["active_cv"] = _safe_div(df["Active Std"], df["Active Mean"])
        added.append("active_cv")

    return added


def _add_eng_flag_density(df: pd.DataFrame) -> list[str]:
    added = []

    if not _has_cols(df, [COL_FWD_PKTS, COL_BWD_PKTS]):
        return added

    total_pkts = df[COL_FWD_PKTS] + df[COL_BWD_PKTS]

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

    if _has_cols(df, ["Fwd PSH Flags", COL_FWD_PKTS]):
        df["fwd_psh_density"] = _safe_div(df["Fwd PSH Flags"], df[COL_FWD_PKTS])
        added.append("fwd_psh_density")

    if _has_cols(df, ["Fwd URG Flags", COL_FWD_PKTS]):
        df["fwd_urg_density"] = _safe_div(df["Fwd URG Flags"], df[COL_FWD_PKTS])
        added.append("fwd_urg_density")

    return added


def _add_eng_flow_indicators(df: pd.DataFrame) -> list[str]:
    added = []

    if COL_BWD_PKTS in df.columns:
        df["is_unidirectional"] = (df[COL_BWD_PKTS] == 0).astype(int)
        added.append("is_unidirectional")

    if "Flow Duration" in df.columns:
        df["is_short_flow"] = (df["Flow Duration"] < 100_000).astype(int)
        added.append("is_short_flow")

    if _has_cols(df, ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]):
        df["init_win_ratio"] = _safe_div(
            df["Init_Win_bytes_forward"].clip(lower=0),
            df["Init_Win_bytes_backward"].clip(lower=0),
        )
        added.append("init_win_ratio")

    if _has_cols(df, ["Subflow Fwd Bytes", "Subflow Bwd Bytes"]):
        df["subflow_bytes_ratio"] = _safe_div(
            df["Subflow Bwd Bytes"], df["Subflow Fwd Bytes"]
        )
        added.append("subflow_bytes_ratio")

    if "Init_Win_bytes_forward" in df.columns:
        df["init_win_fwd_is_zero"] = (df["Init_Win_bytes_forward"] <= 0).astype(int)
        added.append("init_win_fwd_is_zero")

    if _has_cols(df, ["min_seg_size_forward", "Fwd Header Length"]):
        df["fwd_min_seg_to_header"] = _safe_div(
            df["min_seg_size_forward"], df["Fwd Header Length"]
        )
        added.append("fwd_min_seg_to_header")

    return added


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


DOMAIN_FEATURES: dict[str, Callable[[pd.DataFrame], list[str]]] = {
    "Eng_Packet_Shape": _add_eng_packet_shape,
    "Eng_Fwd_Header_Load": _add_eng_fwd_header_load,
    "Eng_Temporal_Burstiness": _add_eng_temporal_burstiness,
    "Eng_Flag_Density": _add_eng_flag_density,
    "Eng_Flow_Indicators": _add_eng_flow_indicators,
    "Eng_Flow_Rates": _add_eng_flow_rates,
}


GROUP_FEATURES: dict[str, list[str]] = {
    "Eng_Packet_Shape": [
        "bytes_ratio_bwd_fwd",
        "pkts_ratio_bwd_fwd",
        "avg_pkt_size_fwd",
        "avg_pkt_size_bwd",
        "pkt_size_ratio_bwd_fwd",
        "pkt_len_cv",
        "pkt_len_min_max_ratio",
    ],
    "Eng_Fwd_Header_Load": [
        "fwd_header_to_payload_ratio",
        "bwd_header_to_payload_ratio",
        "fwd_header_per_pkt",
        "bwd_header_per_pkt",
        "fwd_data_pkt_ratio",
    ],
    "Eng_Temporal_Burstiness": [
        "fwd_iat_cv",
        "bwd_iat_cv",
        "flow_iat_cv",
        "duration_per_pkt",
        "active_idle_ratio",
        "active_cv",
    ],
    "Eng_Flag_Density": [
        "syn_flag_count_density",
        "psh_flag_count_density",
        "ack_flag_count_density",
        "rst_flag_count_density",
        "urg_flag_count_density",
        "fin_flag_count_density",
        "ece_flag_count_density",
        "fwd_psh_density",
        "fwd_urg_density",
    ],
    "Eng_Flow_Indicators": [
        "is_unidirectional",
        "is_short_flow",
        "init_win_ratio",
        "subflow_bytes_ratio",
        "init_win_fwd_is_zero",
        "fwd_min_seg_to_header",
    ],
    "Eng_Flow_Rates": [
        "pkt_rate",
        "fwd_pkt_rate",
        "bwd_pkt_rate",
        "byte_rate",
        "fwd_byte_rate",
    ],
}

FEATURE_TO_GROUP: dict[str, str] = {
    feat: group for group, feats in GROUP_FEATURES.items() for feat in feats
}


def add_domain_features(
    df: pd.DataFrame,
    features: Iterable[str] | None = None,
) -> pd.DataFrame:
    if not features:
        return df

    features = list(features)

    requested_groups: set[str] = set()
    requested_features: set[str] = set()
    for name in features:
        if name in DOMAIN_FEATURES:
            requested_groups.add(name)
        elif name in FEATURE_TO_GROUP:
            requested_features.add(name)
        else:
            logger.warning(
                "[domain_features] Nome desconhecido '%s' — ignorando. "
                "Grupos: %s | Features: %s",
                name,
                list(DOMAIN_FEATURES.keys()),
                list(FEATURE_TO_GROUP.keys()),
            )

    groups_to_run = set(requested_groups)
    for feat in requested_features:
        groups_to_run.add(FEATURE_TO_GROUP[feat])

    if not groups_to_run:
        return df

    logger.info("=" * 70)
    logger.info(
        "[domain_features] Grupos: %s | Features avulsas: %s",
        sorted(requested_groups),
        sorted(requested_features),
    )
    logger.info("[domain_features] Shape de entrada: %s", df.shape)

    df = df.copy()
    n_before = df.shape[1]

    produced: dict[str, list[str]] = {}
    for group in DOMAIN_FEATURES:
        if group in groups_to_run:
            added = DOMAIN_FEATURES[group](df)
            produced[group] = added
            logger.info(
                "[domain_features] %s: %d features computadas (%s)",
                group,
                len(added),
                added,
            )

    keep: set[str] = set(requested_features)
    for group in requested_groups:
        keep.update(produced.get(group, []))

    to_drop = [
        col
        for added in produced.values()
        for col in added
        if col not in keep and col in df.columns
    ]
    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info(
            "[domain_features] Descartadas %d features não solicitadas (%s)",
            len(to_drop),
            to_drop,
        )

    not_produced = requested_features - {
        c for added in produced.values() for c in added
    }
    if not_produced:
        logger.warning(
            "[domain_features] Features pedidas mas não produzidas "
            "(colunas de origem ausentes?): %s",
            sorted(not_produced),
        )

    logger.info(
        "[domain_features] Total: %d novas features. Shape de saída: %s",
        df.shape[1] - n_before,
        df.shape,
    )
    logger.info("=" * 70)

    return df
