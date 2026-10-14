"""Synthetic-data checks only; do not validate historical dissertation scores."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toxicity_baseline import evaluate_models, load_comments


def example_data() -> pd.DataFrame:
    texts = [f"abusive hostile message sample {i}" for i in range(24)]
    texts += [f"friendly helpful message sample {i}" for i in range(24)]
    return pd.DataFrame({"Text": texts, "IsToxic": [1] * 24 + [0] * 24})


def test_works_on_synthetic_data():
    metrics = evaluate_models(example_data(), folds=3, seed=42)
    assert metrics["train_rows"] + metrics["test_rows"] == 48
    assert set(metrics["models"]) == {"LogisticRegression", "SVC", "DecisionTree", "RandomForest", "KNN"}
    for values in metrics["models"].values():
        for name in ("train_cv_accuracy_mean", "test_accuracy", "test_f1", "test_roc_auc"):
            assert 0 <= values[name] <= 1


def test_rejects_missing_columns(tmp_path):
    p = tmp_path / "incorrect.csv"
    pd.DataFrame({"comment": ["hello"], "label": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        load_comments(p)


def test_rejects_invalid_labels(tmp_path):
    p = tmp_path / "invalid.csv"
    pd.DataFrame({"Text": ["a", "b"], "IsToxic": [0, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="binary labels"):
        load_comments(p)
