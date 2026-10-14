"""Tests for the SHAP explainability module (Stage 1 Binary Classifier)."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.models.explainability import (
    compute_shap_explanations,
    extract_local_explanation,
    get_top_features_importance,
)


@pytest.fixture
def trained_binary_model_and_data():
    """Provides a trained small LGBMClassifier and evaluation data."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 8
    feature_names = [f"feat_{i}" for i in range(n_features)]

    X = pd.DataFrame(rng.standard_normal((n_samples, n_features)), columns=feature_names)
    # Target correlated with feat_0 and feat_1
    prob = 1 / (1 + np.exp(-(2.0 * X["feat_0"] - 1.5 * X["feat_1"])))
    y = rng.binomial(1, prob)

    model = LGBMClassifier(n_estimators=15, random_state=42, verbose=-1)
    model.fit(X, y)

    return model, X, y, feature_names


def test_tree_explainer_additivity_property(trained_binary_model_and_data):
    """Test that SHAP values sum + expected_value reconstructs the raw log-odds score."""
    model, X, _, feature_names = trained_binary_model_and_data

    explanation = compute_shap_explanations(model, X, feature_names=feature_names)

    shap_values = explanation["shap_values"]
    expected_value = explanation["expected_value"]
    raw_scores = model.predict_proba(X, raw_score=True)

    reconstructed_scores = shap_values.sum(axis=1) + expected_value

    np.testing.assert_allclose(
        reconstructed_scores,
        raw_scores,
        rtol=1e-5,
        atol=1e-5,
        err_msg="SHAP additivity property violated: sum(phi_i) + phi_0 != raw_score",
    )


def test_top_features_importance(trained_binary_model_and_data):
    """Test that feature importance ranking is correctly sorted by mean absolute SHAP."""
    model, X, _, feature_names = trained_binary_model_and_data

    explanation = compute_shap_explanations(model, X, feature_names=feature_names)
    top_df = get_top_features_importance(explanation["shap_values"], feature_names, top_n=5)

    assert len(top_df) == 5
    assert list(top_df.columns) == ["feature", "mean_abs_shap"]

    # Verify descending sort order
    values = top_df["mean_abs_shap"].to_numpy()
    assert np.all(values[:-1] >= values[1:])

    # Top features should include feat_0 and feat_1 given our data generation
    top_feats = top_df["feature"].tolist()
    assert "feat_0" in top_feats or "feat_1" in top_feats


def test_extract_local_explanation(trained_binary_model_and_data):
    """Test local explanation extraction for a single transaction instance."""
    model, X, y, feature_names = trained_binary_model_and_data

    explanation = compute_shap_explanations(model, X, feature_names=feature_names)

    sample_idx = 10
    local_exp = extract_local_explanation(
        explanation=explanation,
        sample_index=sample_idx,
        X=X,
        y_true=y[sample_idx],
        y_prob=float(model.predict_proba(X.iloc[[sample_idx]])[0, 1]),
    )

    assert local_exp["sample_index"] == sample_idx
    assert "feature_contributions" in local_exp
    assert len(local_exp["feature_contributions"]) == len(feature_names)
    assert "base_value" in local_exp
    assert "y_true" in local_exp
    assert "y_prob" in local_exp
