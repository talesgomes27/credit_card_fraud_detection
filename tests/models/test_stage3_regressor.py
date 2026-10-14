import os

from src.data.loader import load_clean_data
from src.data.splitter import time_ordered_split
from src.features.pipeline import FeaturePipeline
from src.models.stage3_regressor import train_stage3_regressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "credit_card_fraud_detection_2026.csv")


def test_stage3_regressor_training_and_metrics():
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    pipeline = FeaturePipeline()
    X_train_trans = pipeline.fit_transform(splits.X_train)
    X_val_trans = pipeline.transform(splits.X_val)

    # Filter to fraud-only instances
    train_fraud_mask = splits.y_train_bin == 1
    val_fraud_mask = splits.y_val_bin == 1

    X_train_fraud = X_train_trans[train_fraud_mask]
    y_train_fraud = splits.y_train_reg[train_fraud_mask]

    X_val_fraud = X_val_trans[val_fraud_mask]
    y_val_fraud = splits.y_val_reg[val_fraud_mask]

    results = train_stage3_regressor(
        X_train=X_train_fraud,
        y_train=y_train_fraud,
        X_val=X_val_fraud,
        y_val=y_val_fraud,
    )

    # Benchmark requirements: RMSE between 45 and 75 USD, R2 > 0.85
    assert results.val_rmse <= 75.0
    assert results.val_mae <= 30.0
    assert results.val_r2 >= 0.85
