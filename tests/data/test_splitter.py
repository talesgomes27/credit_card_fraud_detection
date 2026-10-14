import os

from src.data.loader import load_clean_data
from src.data.splitter import time_ordered_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "credit_card_fraud_detection_2026.csv")


def test_time_ordered_split_shapes_and_dates():
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    # Check row counts (exact splits by date)
    assert splits.X_train.shape[0] == 66388
    assert splits.X_val.shape[0] == 16905
    assert splits.X_test.shape[0] == 16707
    assert splits.X_train.shape[0] + splits.X_val.shape[0] + splits.X_test.shape[0] == 100000

    # Check temporal boundaries
    assert splits.X_train["transaction_datetime"].max() < "2026-09-01"
    assert splits.X_val["transaction_datetime"].min() >= "2026-09-01"
    assert splits.X_val["transaction_datetime"].max() < "2026-11-01"
    assert splits.X_test["transaction_datetime"].min() >= "2026-11-01"


def test_targets_separated_from_features():
    df = load_clean_data(DATA_PATH)
    splits = time_ordered_split(df)

    targets = ["is_fraud", "fraud_type", "fraud_amount_lost"]
    for target in targets:
        assert target not in splits.X_train.columns
        assert target not in splits.X_val.columns
        assert target not in splits.X_test.columns

    # Check target series lengths
    assert len(splits.y_train_bin) == 66388
    assert len(splits.y_val_bin) == 16905
    assert len(splits.y_test_bin) == 16707

    # Check target types
    assert set(splits.y_train_bin.unique()) == {0, 1}
    assert "Legitimate" in splits.y_train_multi.unique()
    assert (splits.y_train_reg[splits.y_train_bin == 0] == 0.0).all()
