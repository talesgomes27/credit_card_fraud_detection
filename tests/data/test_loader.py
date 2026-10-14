import os

from src.data.loader import load_clean_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "credit_card_fraud_detection_2026.csv")


def test_load_clean_data_shape_and_dropped_ids():
    df = load_clean_data(DATA_PATH)
    assert df.shape[0] == 100000
    assert df.shape[1] == 41
    assert "transaction_id" not in df.columns
    assert "card_id" not in df.columns
    assert "customer_id" in df.columns
    assert "transaction_datetime" in df.columns


def test_browser_na_preserved_as_category():
    df = load_clean_data(DATA_PATH)
    # Zero nulls allowed in browser
    assert df["browser"].isnull().sum() == 0
    # Exactly 41,709 physical in-store and ATM transactions have browser == 'N/A'
    assert (df["browser"] == "N/A").sum() == 41709
    # In-store transactions must have browser == 'N/A'
    in_store = df[df["transaction_channel"] == "In-store"]
    assert (in_store["browser"] == "N/A").all()


def test_datetime_is_strictly_sorted():
    df = load_clean_data(DATA_PATH)
    assert df["transaction_datetime"].is_monotonic_increasing


def test_zero_nulls_across_entire_dataset():
    df = load_clean_data(DATA_PATH)
    assert df.isnull().sum().sum() == 0
