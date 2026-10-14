"""Módulo do Estágio 3: Regressor condicional para estimativa de perda financeira em fraudes (ADRs 0001 e 0003)."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


@dataclass
class Stage3Results:
    """Resultados de treino e avaliação do Estágio 3 (Regressão de Perda Residual)."""

    model: lgb.LGBMRegressor
    val_rmse: float
    val_mae: float
    val_r2: float


def train_stage3_regressor(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> Stage3Results:
    """Treina o regressor LightGBM sobre a partição de fraudes confirmadas

    para quantificar o dano financeiro residual pós-recuperação parcial.
    """
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
    )
    reg.fit(X_train, y_train)

    y_val_pred = np.clip(reg.predict(X_val), 0.0, None)

    rmse = float(root_mean_squared_error(y_val, y_val_pred))
    mae = float(mean_absolute_error(y_val, y_val_pred))
    r2 = float(r2_score(y_val, y_val_pred))

    return Stage3Results(
        model=reg,
        val_rmse=rmse,
        val_mae=mae,
        val_r2=r2,
    )
