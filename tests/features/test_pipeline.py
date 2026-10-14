import os

import numpy as np

from src.data.loader import load_clean_data
from src.data.splitter import time_ordered_split
from src.features.pipeline import FeaturePipeline, engineer_features

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "credit_card_fraud_detection_2026.csv")


def test_engineer_features_creates_correct_columns():
    df = load_clean_data(DATA_PATH).head(100)
    df_feat = engineer_features(df)

    # Check that new engineered columns are present
    expected_cols = [
        "hour_sin",
        "hour_cos",
        "is_night_window",
        "apparent_velocity_kmh",
        "is_impossible_speed",
        "amount_zscore_30d",
        "is_spike_amount",
        "credit_limit_utilization",
        "amount_per_tx_1h",
        "velocity_burst",
    ]
    for col in expected_cols:
        assert col in df_feat.columns

    # Check bounds
    assert (df_feat["hour_sin"] >= -1.0).all() and (df_feat["hour_sin"] <= 1.0).all()
    assert (df_feat["hour_cos"] >= -1.0).all() and (df_feat["hour_cos"] <= 1.0).all()
    assert set(df_feat["is_night_window"].unique()).issubset({0, 1})
    assert (df_feat["apparent_velocity_kmh"] >= 0.0).all()


def test_feature_pipeline_fit_transform():
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    pipeline = FeaturePipeline()
    X_train_trans = pipeline.fit_transform(splits.X_train)
    X_val_trans = pipeline.transform(splits.X_val)
    X_test_trans = pipeline.transform(splits.X_test)

    # Assert matrix dimensions and consistency
    assert X_train_trans.shape[0] == 66388
    assert X_val_trans.shape[0] == 16905
    assert X_test_trans.shape[0] == 16707

    assert X_train_trans.shape[1] == X_val_trans.shape[1]
    assert X_train_trans.shape[1] == X_test_trans.shape[1]

    # Zero NaNs allowed in final transformed matrices
    assert np.isnan(X_train_trans).sum() == 0
    assert np.isnan(X_val_trans).sum() == 0
    assert np.isnan(X_test_trans).sum() == 0

    # Ensure feature names are accessible
    feature_names = pipeline.get_feature_names_out()
    assert len(feature_names) == X_train_trans.shape[1]
    assert "apparent_velocity_kmh" in feature_names
