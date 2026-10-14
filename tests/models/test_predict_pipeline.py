import os
import tempfile

import pandas as pd

from src.data.loader import load_clean_data
from src.data.splitter import time_ordered_split
from src.models.predict import FraudDetectionHurdlePipeline

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "credit_card_fraud_detection_2026.csv")


def test_hurdle_pipeline_end_to_end():
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    # Instantiate and fit hurdle pipeline
    pipeline = FraudDetectionHurdlePipeline()
    pipeline.fit(
        splits.X_train,
        splits.y_train_bin,
        splits.y_train_multi,
        splits.y_train_reg,
        splits.X_val,
        splits.y_val_bin,
        splits.y_val_multi,
        splits.y_val_reg,
    )

    # Predict on test set
    preds_df = pipeline.predict_transactions(splits.X_test)

    # Check output shape and columns
    assert preds_df.shape[0] == 16707
    expected_cols = [
        "fraud_probability",
        "is_fraud_pred",
        "predicted_fraud_type",
        "predicted_amount_lost",
    ]
    for col in expected_cols:
        assert col in preds_df.columns

    # Verify Hurdle invariant rules:
    # 1. Legitimate predicted instances MUST have type 'Legitimate' and amount 0.0
    legit_mask = preds_df["is_fraud_pred"] == 0
    assert (preds_df.loc[legit_mask, "predicted_fraud_type"] == "Legitimate").all()
    assert (preds_df.loc[legit_mask, "predicted_amount_lost"] == 0.0).all()

    # 2. Fraud predicted instances MUST have non-legitimate type and amount >= 0.0
    fraud_mask = preds_df["is_fraud_pred"] == 1
    if fraud_mask.sum() > 0:
        assert (preds_df.loc[fraud_mask, "predicted_fraud_type"] != "Legitimate").all()
        assert (preds_df.loc[fraud_mask, "predicted_amount_lost"] >= 0.0).all()

    # Test persistence (save and load)
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        pipeline.save(tmp.name)
        loaded = FraudDetectionHurdlePipeline.load(tmp.name)
        loaded_preds = loaded.predict_transactions(splits.X_test.head(10))
        pd.testing.assert_frame_equal(preds_df.head(10), loaded_preds)
