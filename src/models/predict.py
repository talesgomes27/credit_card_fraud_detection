"""Orquestrador do Hurdle Model em três estágios para predição ponta a ponta (ADR 0001 e Gate 3)."""

from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.models.stage1_binary import train_stage1_model
from src.models.stage2_multiclass import train_stage2_multiclass
from src.models.stage3_regressor import train_stage3_regressor


class FraudDetectionHurdlePipeline:
    """Pipeline unificado que integra pré-processamento e os três estágios preditivos:

    Estágio 1: is_fraud (Classificação binária com threshold de custo calibrado)
    Estágio 2: fraud_type (Classificação multiclasse condicional)
    Estágio 3: fraud_amount_lost (Regressão condicional da perda em dólares)
    """

    def __init__(self) -> None:
        self.feature_pipeline = FeaturePipeline()
        self.stage1_model = None
        self.best_threshold: float = 0.50
        self.stage2_model = None
        self.stage2_classes: list[str] = []
        self.stage3_model = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_bin: pd.Series,
        y_train_multi: pd.Series,
        y_train_reg: pd.Series,
        X_val: pd.DataFrame,
        y_val_bin: pd.Series,
        y_val_multi: pd.Series,
        y_val_reg: pd.Series,
    ) -> Self:
        """Ajusta o pipeline de features e treina os três modelos nos dados de treino e validação."""
        # 1. Ajuste do pipeline de features
        X_train_trans = self.feature_pipeline.fit_transform(X_train)
        X_val_trans = self.feature_pipeline.transform(X_val)

        # 2. Treinamento do Estágio 1 (Detecção binária e tuning de custo)
        stage1_res = train_stage1_model(
            X_train=X_train_trans,
            y_train=y_train_bin,
            X_val=X_val_trans,
            y_val=y_val_bin,
            val_lost_amounts=y_val_reg,
        )
        self.stage1_model = stage1_res.model
        self.best_threshold = stage1_res.best_threshold

        # 3. Treinamento do Estágio 2 (Tipificação multiclasse sobre fraudes)
        train_fraud_mask = y_train_bin == 1
        val_fraud_mask = y_val_bin == 1

        stage2_res = train_stage2_multiclass(
            X_train=X_train_trans[train_fraud_mask],
            y_train=y_train_multi[train_fraud_mask],
            X_val=X_val_trans[val_fraud_mask],
            y_val=y_val_multi[val_fraud_mask],
        )
        self.stage2_model = stage2_res.model
        self.stage2_classes = stage2_res.classes

        # 4. Treinamento do Estágio 3 (Regressão de dano sobre fraudes)
        stage3_res = train_stage3_regressor(
            X_train=X_train_trans[train_fraud_mask],
            y_train=y_train_reg[train_fraud_mask],
            X_val=X_val_trans[val_fraud_mask],
            y_val=y_val_reg[val_fraud_mask],
        )
        self.stage3_model = stage3_res.model

        return self

    def predict_transactions(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Processa novas transações e retorna DataFrame enriquecido com probabilidades,

        decisões binárias, tipificação de ataque e estimativa de perda monetária.
        """
        X_trans = self.feature_pipeline.transform(df_raw)

        # Estágio 1: Probabilidade e decisão binária
        probs = self.stage1_model.predict_proba(X_trans)[:, 1]
        is_fraud_preds = (probs >= self.best_threshold).astype(int)

        n_rows = len(df_raw)
        pred_types = np.array(["Legitimate"] * n_rows, dtype=object)
        pred_amounts = np.zeros(n_rows, dtype=float)

        fraud_indices = np.where(is_fraud_preds == 1)[0]
        if len(fraud_indices) > 0:
            X_trans_fraud = X_trans[fraud_indices]

            # Estágio 2: Tipificação
            types_pred = self.stage2_model.predict(X_trans_fraud)
            pred_types[fraud_indices] = types_pred

            # Estágio 3: Estimativa de perda
            amounts_pred = np.clip(self.stage3_model.predict(X_trans_fraud), 0.0, None)
            pred_amounts[fraud_indices] = amounts_pred

        return pd.DataFrame(
            {
                "fraud_probability": probs,
                "is_fraud_pred": is_fraud_preds,
                "predicted_fraud_type": pred_types,
                "predicted_amount_lost": pred_amounts,
            },
            index=df_raw.index,
        )

    def save(self, filepath: str | Path) -> None:
        """Persiste o pipeline completo em disco via joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str | Path) -> "FraudDetectionHurdlePipeline":
        """Carrega o pipeline serializado do disco."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline não encontrado em: {path}")
        return joblib.load(path)
