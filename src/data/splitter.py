"""Módulo de particionamento temporal estrito out-of-time (ADR 0002)."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSplits:
    """Estrutura imutável contendo as partições temporais de treino, validação e teste cego."""

    X_train: pd.DataFrame
    y_train_bin: pd.Series
    y_train_multi: pd.Series
    y_train_reg: pd.Series

    X_val: pd.DataFrame
    y_val_bin: pd.Series
    y_val_multi: pd.Series
    y_val_reg: pd.Series

    X_test: pd.DataFrame
    y_test_bin: pd.Series
    y_test_multi: pd.Series
    y_test_reg: pd.Series


def time_ordered_split(
    df: pd.DataFrame,
    train_end: str = "2026-09-01",
    val_end: str = "2026-11-01",
) -> DataSplits:
    """Divide a base em Treino (Jan-Ago), Validação (Set-Out) e Teste Cego (Nov-Dez)

    assegurando isolamento total entre matriz de atributos e variáveis target.
    """
    target_cols = ["is_fraud", "fraud_type", "fraud_amount_lost"]

    train_mask = df["transaction_datetime"] < train_end
    val_mask = (df["transaction_datetime"] >= train_end) & (df["transaction_datetime"] < val_end)
    test_mask = df["transaction_datetime"] >= val_end

    df_train = df[train_mask].reset_index(drop=True)
    df_val = df[val_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in target_cols]

    return DataSplits(
        X_train=df_train[feature_cols].copy(),
        y_train_bin=df_train["is_fraud"].copy(),
        y_train_multi=df_train["fraud_type"].copy(),
        y_train_reg=df_train["fraud_amount_lost"].copy(),
        X_val=df_val[feature_cols].copy(),
        y_val_bin=df_val["is_fraud"].copy(),
        y_val_multi=df_val["fraud_type"].copy(),
        y_val_reg=df_val["fraud_amount_lost"].copy(),
        X_test=df_test[feature_cols].copy(),
        y_test_bin=df_test["is_fraud"].copy(),
        y_test_multi=df_test["fraud_type"].copy(),
        y_test_reg=df_test["fraud_amount_lost"].copy(),
    )
