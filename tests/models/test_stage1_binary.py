import os

import pytest

from src.data.loader import load_clean_data
from src.data.splitter import time_ordered_split
from src.features.pipeline import FeaturePipeline
from src.models.stage1_binary import evaluate_financial_cost, train_stage1_model

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "credit_card_fraud_detection_2026.csv")


def test_evaluate_financial_cost_calculation():
    # Synthetic test of cost logic
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.8, 0.2, 0.9]
    real_lost = [0.0, 0.0, 100.0, 200.0]

    # At threshold 0.5:
    # row 0: prob 0.1 -> pred 0 (TN, cost 0)
    # row 1: prob 0.8 -> pred 1 (FP, cost 3.50)
    # row 2: prob 0.2 -> pred 0 (FN, cost 100.0)
    # row 3: prob 0.9 -> pred 1 (TP, cost 0)
    # Expected cost = 100.0 + 3.50 = 103.50
    cost, fp_count, fn_count = evaluate_financial_cost(y_true, y_prob, real_lost, threshold=0.5, cost_fp=3.50)
    assert cost == pytest.approx(103.50, rel=1e-3)
    assert fp_count == 1
    assert fn_count == 1


def test_stage1_binary_training_and_metrics():
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    pipeline = FeaturePipeline()
    X_train_trans = pipeline.fit_transform(splits.X_train)
    X_val_trans = pipeline.transform(splits.X_val)

    results = train_stage1_model(
        X_train=X_train_trans,
        y_train=splits.y_train_bin,
        X_val=X_val_trans,
        y_val=splits.y_val_bin,
        val_lost_amounts=splits.y_val_reg,
    )

    # Benchmark requirements:
    # Expected ROC-AUC around 0.84-0.85
    assert results.val_roc_auc >= 0.80
    # PR-AUC significantly above prior prevalence (0.02)
    assert results.val_pr_auc >= 0.12
    # Optimal threshold is within sane bounds
    assert 0.01 <= results.best_threshold <= 0.80
    assert len(results.cost_curve) > 0
