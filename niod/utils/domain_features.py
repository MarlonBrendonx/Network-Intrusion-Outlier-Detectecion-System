"""
Feature engineering de domínio para detecção de intrusão de rede.

Cada função adiciona uma feature derivada de razões entre colunas existentes,
projetada para evidenciar padrões específicos de tráfego anômalo. A ideia é
que essas razões são invariantes a escala (um ataque de 10 ou 10.000 pacotes
pode ter a mesma "assinatura" proporcional), o que ajuda generalização.

As features são opt-in: só são adicionadas quando explicitamente pedidas
via parâmetro `domain_features` em `prepare_splits` ou pela flag
`--domain-features` na CLI.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPSILON: float = 1e-6

# Mapeia nome curto da feature → (colunas requeridas, função de cálculo)
# Cada entrada produz UMA coluna nova no DataFrame.


def _eng_packet_shape(df: pd.DataFrame) -> pd.Series:
    """
    Eng_Packet_Shape = Fwd Packet Length Min / (Fwd Packet Length Max + ε)

    Detecta scripts de automação que geram pacotes de tamanho idêntico:
    quando Min ≈ Max, a razão tende a 1, indicando uniformidade artificial.
    Tráfego humano legítimo tem variação maior, então Min << Max e a
    razão fica próxima de 0.
    """
    return df["Fwd Packet Length Min"] / (df["Fwd Packet Length Max"] + EPSILON)


def _eng_fwd_header_load(df: pd.DataFrame) -> pd.Series:
    """
    Eng_Fwd_Header_Load = Fwd Header Length / (Subflow Fwd Bytes + ε)

    Expõe ataques com muito cabeçalho e pouca carga útil:
    valores altos indicam que o overhead de protocolo é desproporcional
    aos dados transferidos — característico de port scans (cabeçalhos
    SYN sem payload), reconhecimento e probes.
    """
    return df["Fwd Header Length"] / (df["Subflow Fwd Bytes"] + EPSILON)


# Registry: nome -> (colunas necessárias, função)
DOMAIN_FEATURES: dict[str, tuple[list[str], callable]] = {
    "Eng_Packet_Shape": (
        ["Fwd Packet Length Min", "Fwd Packet Length Max"],
        _eng_packet_shape,
    ),
    "Eng_Fwd_Header_Load": (
        ["Fwd Header Length", "Subflow Fwd Bytes"],
        _eng_fwd_header_load,
    ),
}


def add_domain_features(
    df: pd.DataFrame,
    *,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Adiciona features de engenharia de domínio ao DataFrame.

    Args:
        df: DataFrame com as colunas brutas do CIC-IDS2017.
        features: Lista de nomes a adicionar. Se None, adiciona TODAS
                  as registradas em DOMAIN_FEATURES. Nomes inválidos
                  são ignorados com aviso.

    Returns:
        DataFrame com as colunas novas adicionadas. Retorna uma cópia
        — não modifica o input.
    """
    df = df.copy()

    if features is None:
        features = list(DOMAIN_FEATURES.keys())

    added = []
    skipped = []

    for name in features:
        if name not in DOMAIN_FEATURES:
            logger.warning("Feature de domínio desconhecida: %r. Ignorando.", name)
            continue

        required_cols, fn = DOMAIN_FEATURES[name]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(
                "Feature %r requer colunas ausentes: %s. Pulando.",
                name,
                missing,
            )
            skipped.append(name)
            continue

        df[name] = fn(df)
        added.append(name)

    # Tratar inf/NaN gerados pela divisão (caso ε não seja suficiente)
    if added:
        df[added] = df[added].replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(
        "Feature engineering de domínio: %d adicionadas %s%s",
        len(added),
        added,
        f" | %d puladas" % len(skipped) if skipped else "",
    )

    return df
