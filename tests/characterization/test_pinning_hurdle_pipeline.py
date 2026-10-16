"""Testes de Caracterização (Pinning Tests / Golden Master) para o Hurdle Model.

Congela o comportamento real do pipeline em 3 estágios, splits e invariantes de negócio
antes de qualquer alteração estrutural no código ou no relatório Quarto.
"""

from pathlib import Path

import numpy as np

from src.data.loader import load_clean_data
from src.data.splitter import time_ordered_split
from src.visualization.figures import EXPECTED_FIGURE_FILENAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "credit_card_fraud_detection_2026.csv"


def test_pinning_temporal_split_golden_invariants():
    """Congela dimensões exatas, janelas cronológicas e contagem de fraudes reais."""
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    # Invariantes de Partição (Total = 100.000 transações)
    assert splits.X_train.shape == (66388, 38)
    assert splits.X_val.shape == (16905, 38)
    assert splits.X_test.shape == (16707, 38)

    # Invariantes de Alvo Binário (Total = 2.055 fraudes)
    assert int(splits.y_train_bin.sum()) == 1341
    assert int(splits.y_val_bin.sum()) == 362
    assert int(splits.y_test_bin.sum()) == 352

    # Invariantes de Alvo Multiclasse
    expected_multiclass = {
        "Legitimate",
        "Card-Not-Present",
        "Account-Takeover",
        "Synthetic-Identity",
        "Card-Present",
    }
    assert set(splits.y_train_multi.unique()) == expected_multiclass

    # Invariantes de Alvo Contínuo (Perda Monetária)
    # Transações legítimas devem possuir perda exatamente igual a 0.0
    legit_train = splits.y_train_bin == 0
    assert np.allclose(splits.y_train_reg[legit_train], 0.0)

    # Fraudes no treino devem ter média histórica de perda dentro da faixa esperada
    fraud_train = splits.y_train_bin == 1
    mean_fraud_loss = float(splits.y_train_reg[fraud_train].mean())
    assert 130.0 <= mean_fraud_loss <= 145.0


def test_pinning_hurdle_logic_invariants():
    """Congela as regras lógicas invariantes do Hurdle Model em 3 estágios."""
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    # Verifica isolamento estrito entre features e targets
    target_columns = ["is_fraud", "fraud_type", "fraud_amount_lost"]
    for col in target_columns:
        assert col not in splits.X_train.columns
        assert col not in splits.X_val.columns
        assert col not in splits.X_test.columns

    # Verifica integridade da coluna browser (preservação literal de 'N/A' como string limpa)
    assert "N/A" in set(splits.X_train["browser"].unique())
    assert splits.X_train["browser"].isna().sum() == 0
    assert (splits.X_train["browser"] == "N/A").sum() == 27617


def test_pinning_visual_artifacts_inventory():
    """Garante a existência e integridade do inventário das 8 figuras canônicas."""
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    assert figures_dir.exists(), "Diretório reports/figures deve existir."

    for filename in EXPECTED_FIGURE_FILENAMES:
        fig_path = figures_dir / filename
        assert fig_path.exists(), f"Figura canônica {filename} ausente do inventário."
        assert fig_path.stat().st_size > 10_000, f"Figura {filename} está corrompida ou vazia."
