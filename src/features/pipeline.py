"""Módulo de engenharia de atributos e pipeline de transformações (ADR 0003 e Gate 3)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera atributos derivados de domínio a partir das colunas originais."""
    df_feat = df.copy()

    # Decomposição periódica da hora do dia
    hour = df_feat["transaction_hour"].astype(float)
    df_feat["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df_feat["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df_feat["is_night_window"] = df_feat["transaction_hour"].isin([1, 2, 3, 4]).astype(int)

    # Dinâmica espacial e velocidade de deslocamento (km/h)
    time_gap_hours = df_feat["days_since_last_transaction"] * 24.0 + 0.1
    velocity = df_feat["distance_from_last_tx_km"] / time_gap_hours
    df_feat["apparent_velocity_kmh"] = velocity
    df_feat["is_impossible_speed"] = (velocity > 500.0).astype(int)

    # Anomalia de ticket e utilização de limite
    denom_std = df_feat["std_transaction_amount_30d"] + 1.0
    df_feat["amount_zscore_30d"] = (
        df_feat["transaction_amount"] - df_feat["avg_transaction_amount_30d"]
    ) / denom_std
    df_feat["is_spike_amount"] = (df_feat["amount_vs_avg_ratio"] >= 5.0).astype(int)

    credit_lim = np.maximum(df_feat["credit_limit"], 1.0)
    df_feat["credit_limit_utilization"] = df_feat["transaction_amount"] / credit_lim

    # Agregações de velocidade e intensidade transacional
    df_feat["amount_per_tx_1h"] = df_feat["amount_last_1h"] / (
        df_feat["transactions_last_1h"] + 1.0
    )
    df_feat["velocity_burst"] = (df_feat["transactions_last_1h"] >= 3).astype(int)

    return df_feat


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Pipeline que aplica engenharia de atributos e codificação categórica One-Hot

    com suporte completo a fit no treino e transform determinístico em validação/teste.
    """

    CATEGORICAL_COLS = [
        "customer_gender",
        "customer_state",
        "merchant_category",
        "merchant_country",
        "transaction_channel",
        "device_type",
        "browser",
        "email_domain",
        "transaction_day_of_week",
    ]

    DROP_COLS = [
        "customer_id",
        "merchant_id",
        "transaction_datetime",
        "transaction_id",
        "card_id",
        "is_fraud",
        "fraud_type",
        "fraud_amount_lost",
    ]

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.numeric_cols_: list[str] = []
        self.feature_names_out_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeaturePipeline":
        """Ajusta o codificador de categorias restrito aos dados de treino."""
        X_eng = engineer_features(X)

        cols_to_drop = [c for c in self.DROP_COLS if c in X_eng.columns]
        X_clean = X_eng.drop(columns=cols_to_drop)

        self.numeric_cols_ = [c for c in X_clean.columns if c not in self.CATEGORICAL_COLS]

        # Ajusta o OneHotEncoder nas variáveis categóricas
        self.encoder.fit(X_clean[self.CATEGORICAL_COLS])
        encoded_names = list(self.encoder.get_feature_names_out(self.CATEGORICAL_COLS))

        self.feature_names_out_ = self.numeric_cols_ + encoded_names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforma novos dados preservando a ordem e estrutura aprendidas no treino."""
        X_eng = engineer_features(X)

        cols_to_drop = [c for c in self.DROP_COLS if c in X_eng.columns]
        X_clean = X_eng.drop(columns=cols_to_drop)

        num_arr = X_clean[self.numeric_cols_].to_numpy(dtype=np.float64)
        cat_arr = self.encoder.transform(X_clean[self.CATEGORICAL_COLS])

        return np.hstack([num_arr, cat_arr])

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        """Retorna os nomes de todas as colunas transformadas."""
        return self.feature_names_out_
