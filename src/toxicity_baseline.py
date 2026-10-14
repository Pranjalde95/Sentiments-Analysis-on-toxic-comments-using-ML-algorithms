"""Reproducible *new* classical ML baseline for binary toxic-comment detection.

This is not the original MSc experiment and does not reproduce its archived scores.
The input CSV is supplied by the researcher and must contain Text and IsToxic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def load_comments(path: str | Path) -> pd.DataFrame:
    """Read and validate the expected binary classification columns."""
    data = pd.read_csv(path)
    required = {"Text", "IsToxic"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required dataset column(s): {', '.join(sorted(missing))}")

    data = data.loc[:, ["Text", "IsToxic"]].dropna().copy()
    data["Text"] = data["Text"].astype(str).str.strip()
    data = data.loc[data["Text"].ne("")].copy()
    data["IsToxic"] = pd.to_numeric(data["IsToxic"], errors="raise")
    labels = set(data["IsToxic"].unique())
    if labels != {0, 1}:
        raise ValueError(f"IsToxic must have both binary labels 0 and 1; found {sorted(labels)}")
    data["IsToxic"] = data["IsToxic"].astype(int)
    return data.reset_index(drop=True)


def build_models(seed: int) -> dict[str, object]:
    """Classical estimators explored in the historical notebook, with fixed defaults.

    These parameter choices are a *new* baseline, not tuned dissertation settings.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "SVC": SVC(kernel="linear", probability=True, random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def evaluate_models(
    data: pd.DataFrame,
    *,
    seed: int = 42,
    test_size: float = 0.30,
    folds: int = 5,
) -> dict[str, object]:
    """Fit TF-IDF inside CV folds; evaluate a single held-out test split."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not {"Text", "IsToxic"}.issubset(data.columns):
        raise ValueError("Data must include Text and IsToxic columns")

    x_train, x_test, y_train, y_test = train_test_split(
        data["Text"], data["IsToxic"],
        test_size=test_size, random_state=seed, stratify=data["IsToxic"],
    )
    if int(y_train.value_counts().min()) < folds:
        raise ValueError("Each training class must contain at least 'folds' examples")
    if int(y_test.nunique()) != 2:
        raise ValueError("The test set must contain both classes to compute ROC-AUC")

    cross_validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results: dict[str, object] = {
        "dataset_rows_after_missing_value_filter": len(data),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "split_seed": seed,
        "test_size_fraction": test_size,
        "cross_validation_folds": folds,
        "note": "New baseline, not a reproduction of dissertation notebook metrics.",
        "models": {},
    }
    for name, classifier in build_models(seed).items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", classifier),
        ])
        # Refitting the whole pipeline in each fold avoids TF-IDF data leakage.
        cv_scores = cross_val_score(
            pipeline, x_train, y_train, cv=cross_validation, scoring="accuracy", n_jobs=1,
        )
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, 1]
        results["models"][name] = {
            "train_cv_accuracy_mean": float(np.mean(cv_scores)),
            "train_cv_accuracy_std": float(np.std(cv_scores)),
            "test_accuracy": float(accuracy_score(y_test, prediction)),
            "test_f1": float(f1_score(y_test, prediction, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, probability)),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Local CSV with Text and IsToxic")
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.30)
    args = parser.parse_args()

    results = evaluate_models(load_comments(args.data), seed=args.seed, folds=args.folds,
                              test_size=args.test_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved freshly computed baseline metrics to {args.output}")


if __name__ == "__main__":
    main()
