"""Módulo de ingestão e saneamento dos dados brutos de transações."""

from pathlib import Path

import pandas as pd


def load_clean_data(filepath: str | Path) -> pd.DataFrame:
    """Carrega o dataset preservando 'N/A' como categoria válida,

    ordena cronologicamente e descarta identificadores puramente redundantes.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")

    # Leitura com keep_default_na=False para preservar 'N/A' na coluna browser (ADR 0004)
    df = pd.read_csv(path, keep_default_na=False)

    # Ordenação cronológica estável para corrigir inconsistências temporais pontuais
    df = df.sort_values(by="transaction_datetime", ascending=True).reset_index(drop=True)

    # Remoção de identificadores sem valor generalizável
    columns_to_drop = [col for col in ["transaction_id", "card_id"] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df
