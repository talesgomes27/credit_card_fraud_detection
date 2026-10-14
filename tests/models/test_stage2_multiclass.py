import os

from src.data.loader import load_clean_data
from src.data.splitter import time_ordered_split
from src.features.pipeline import FeaturePipeline
from src.models.stage2_multiclass import train_stage2_multiclass

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "credit_card_fraud_detection_2026.csv")


def test_stage2_multiclass_training_and_metrics():
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    pipeline = FeaturePipeline()
    X_train_trans = pipeline.fit_transform(splits.X_train)
    X_val_trans = pipeline.transform(splits.X_val)

    # Filter to fraud-only instances in train and val
    train_fraud_mask = splits.y_train_bin == 1
    val_fraud_mask = splits.y_val_bin == 1

    X_train_fraud = X_train_trans[train_fraud_mask]
    y_train_fraud = splits.y_train_multi[train_fraud_mask]

    X_val_fraud = X_val_trans[val_fraud_mask]
    y_val_fraud = splits.y_val_multi[val_fraud_mask]

    results = train_stage2_multiclass(
        X_train=X_train_fraud,
        y_train=y_train_fraud,
        X_val=X_val_fraud,
        y_val=y_val_fraud,
    )

    # Validate output structure
    expected_classes = ["Account-Takeover", "Card-Not-Present", "Card-Present", "Synthetic-Identity"]
    assert set(results.classes) == set(expected_classes)
    assert results.val_macro_f1 >= 0.40
    assert results.val_balanced_acc >= 0.40
    assert results.confusion_matrix.shape == (4, 4)
