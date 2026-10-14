"""SHAP tree explainability module for Stage 1 binary fraud classification.

Calculates global feature attribution and local waterfall contributions using
TreeExplainer on gradient boosted trees.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier


def compute_shap_explanations(
    model: LGBMClassifier,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    """Computes SHAP values, expected baseline, and top features importance.

    Parameters
    ----------
    model : LGBMClassifier
        Trained LightGBM binary classifier.
    X : pd.DataFrame or np.ndarray
        Feature matrix to explain.
    feature_names : list[str] | None, optional
        Names of the features. Extracted from X if DataFrame.
    top_n : int, default 15
        Number of top features to include in the summary table.

    Returns
    -------
    dict[str, Any]
        Dictionary containing explainer, shap_values array, base expected_value,
        resolved feature names, and top features DataFrame.
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            resolved_feature_names = list(X.columns)
        else:
            resolved_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        resolved_feature_names = list(feature_names)

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X)

    if isinstance(raw_shap, list):
        # In binary classification, index 1 corresponds to positive (fraud) class
        shap_values = raw_shap[1] if len(raw_shap) == 2 else raw_shap[0]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3 and raw_shap.shape[2] == 2:
        shap_values = raw_shap[:, :, 1]
    else:
        shap_values = np.asarray(raw_shap)

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_value = float(expected_val[1] if len(expected_val) == 2 else expected_val[0])
    else:
        expected_value = float(expected_val)

    top_features = get_top_features_importance(
        shap_values=shap_values,
        feature_names=resolved_feature_names,
        top_n=top_n,
    )

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "expected_value": expected_value,
        "feature_names": resolved_feature_names,
        "top_features": top_features,
    }


def get_top_features_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Ranks features by mean absolute SHAP value across all evaluated samples.

    Parameters
    ----------
    shap_values : np.ndarray
        Array of shape (n_samples, n_features).
    feature_names : list[str]
        List of feature names matching columns in shap_values.
    top_n : int, default 15
        Number of highest-ranking features to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    })
    df = df.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    return df.head(min(top_n, len(df)))


def extract_local_explanation(
    explanation: dict[str, Any],
    sample_index: int,
    X: pd.DataFrame | np.ndarray,
    y_true: int | None = None,
    y_prob: float | None = None,
) -> dict[str, Any]:
    """Extracts attribution details for a specific individual transaction.

    Parameters
    ----------
    explanation : dict[str, Any]
        Output dict from compute_shap_explanations.
    sample_index : int
        Row index within the evaluation dataset.
    X : pd.DataFrame or np.ndarray
        Original feature matrix used in explanation.
    y_true : int | None, optional
        Observed ground truth label (0 or 1).
    y_prob : float | None, optional
        Model predicted probability of fraud.

    Returns
    -------
    dict[str, Any]
        Detailed attribution dictionary for the sample.
    """
    shap_values = explanation["shap_values"]
    feature_names = explanation["feature_names"]
    base_val = explanation["expected_value"]

    sample_shap = shap_values[sample_index]

    if isinstance(X, pd.DataFrame):
        sample_vals = X.iloc[sample_index].to_dict()
    else:
        sample_vals = {name: X[sample_index, idx] for idx, name in enumerate(feature_names)}

    contributions = [
        {
            "feature": name,
            "value": sample_vals.get(name),
            "shap_value": float(sample_shap[idx]),
        }
        for idx, name in enumerate(feature_names)
    ]
    # Sort contributions by magnitude
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    return {
        "sample_index": sample_index,
        "base_value": base_val,
        "y_true": y_true,
        "y_prob": y_prob,
        "feature_contributions": contributions,
    }
