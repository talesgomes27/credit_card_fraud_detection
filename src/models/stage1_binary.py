"""Módulo do Estágio 1: Classificador binário para is_fraud e otimização de custo (ADRs 0001, 0003 e 0005)."""

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def evaluate_financial_cost(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    real_lost: Sequence[float],
    threshold: float = 0.50,
    cost_fp: float = 3.50,
) -> tuple[float, int, int]:
    """Calcula o custo financeiro total decorrente de falsos negativos (perda monetária)

    e falsos positivos (custo de fricção/atrito de atendimento).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    lost_arr = np.asarray(real_lost, dtype=float)

    preds = (y_prob_arr >= threshold).astype(int)

    fp_mask = (preds == 1) & (y_true_arr == 0)
    fn_mask = (preds == 0) & (y_true_arr == 1)

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())

    total_cost = float(lost_arr[fn_mask].sum() + fp_count * cost_fp)
    return total_cost, fp_count, fn_count


@dataclass
class Stage1Results:
    """Resultados de treino, calibração e avaliação do Estágio 1."""

    model: lgb.LGBMClassifier
    best_threshold: float
    val_pr_auc: float
    val_roc_auc: float
    val_precision: float
    val_recall: float
    val_f1: float
    min_cost: float
    cost_curve: pd.DataFrame


def train_stage1_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    val_lost_amounts: pd.Series,
    cost_fp: float = 3.50,
    scale_weight: float | None = None,
) -> Stage1Results:
    """Treina o LightGBM ponderado para detecção de fraude e calibra o threshold ótimo

    minimizando a função de custo financeiro na validação.
    """
    if scale_weight is None:
        scale_weight = 5.0

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        scale_pos_weight=scale_weight,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(X_train, y_train)

    y_val_prob = clf.predict_proba(X_val)[:, 1]

    val_pr_auc = float(average_precision_score(y_val, y_val_prob))
    val_roc_auc = float(roc_auc_score(y_val, y_val_prob))

    thresholds = np.linspace(0.01, 0.99, 99)
    curve_records = []

    best_thresh = 0.50
    min_cost = float("inf")
    best_p, best_r, best_f1 = 0.0, 0.0, 0.0

    for thresh in thresholds:
        cost, fp, fn = evaluate_financial_cost(
            y_true=y_val,
            y_prob=y_val_prob,
            real_lost=val_lost_amounts,
            threshold=thresh,
            cost_fp=cost_fp,
        )
        preds = (y_val_prob >= thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0
        )

        curve_records.append(
            {
                "threshold": float(thresh),
                "cost": cost,
                "fp": fp,
                "fn": fn,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }
        )

        if cost < min_cost:
            min_cost = cost
            best_thresh = float(thresh)
            best_p, best_r, best_f1 = float(p), float(r), float(f1)

    cost_df = pd.DataFrame(curve_records)

    return Stage1Results(
        model=clf,
        best_threshold=best_thresh,
        val_pr_auc=val_pr_auc,
        val_roc_auc=val_roc_auc,
        val_precision=best_p,
        val_recall=best_r,
        val_f1=best_f1,
        min_cost=min_cost,
        cost_curve=cost_df,
    )
