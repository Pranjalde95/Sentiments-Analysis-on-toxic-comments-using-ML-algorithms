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


def _normalized_text(text: str) -> str:
    """Normalize only for duplicate detection; retain original text for TF-IDF."""
    return " ".join(text.split()).casefold()


def load_comments(path: str | Path) -> pd.DataFrame:
    """Validate binary labels and remove normalized duplicate texts before splitting.

    Conflicting toxicity labels on normalized-identical text raise an error rather
    than silently choosing one label. Missing/blank texts are excluded.
    """
    raw = pd.read_csv(path)
    required = {"Text", "IsToxic"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required dataset column(s): {', '.join(sorted(missing))}")

    data = raw.loc[:, ["Text", "IsToxic"]].dropna().copy()
    data["Text"] = data["Text"].astype(str).str.strip()
    data = data.loc[data["Text"].ne("")].copy()
    data["IsToxic"] = pd.to_numeric(data["IsToxic"], errors="raise")
    labels = set(data["IsToxic"].unique())
    if labels != {0, 1}:
        raise ValueError(f"IsToxic must have both binary labels 0 and 1; found {sorted(labels)}")
    data["IsToxic"] = data["IsToxic"].astype(int)

    keys = data["Text"].map(_normalized_text)
    conflicting = data.groupby(keys)["IsToxic"].nunique().gt(1)
    if conflicting.any():
        raise ValueError(
            f"Conflicting IsToxic labels found for {int(conflicting.sum())} "
            "normalized-identical comment(s); review labels before deduplication."
        )

    before_deduplication = len(data)
    data = data.loc[~keys.duplicated(keep="first")].reset_index(drop=True)
    data.attrs["input_rows"] = len(raw)
    data.attrs["filtered_missing_or_blank_rows"] = len(raw) - before_deduplication
    data.attrs["duplicate_rows_removed"] = before_deduplication - len(data)
    return data

def build_models(seed: int) -> dict[str, object]:
    """Fixed *new baseline* settings, not historical dissertation tuning."""
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
    """Fit TF-IDF inside CV folds and evaluate a held-out, aligned test split."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not {"Text", "IsToxic"}.issubset(data.columns):
        raise ValueError("Data must include Text and IsToxic columns")
    if data["Text"].astype(str).map(_normalized_text).duplicated().any():
        raise ValueError("Duplicate normalized comment texts remain; first use load_comments().")

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
        "input_rows": int(data.attrs.get("input_rows", len(data))),
        "filtered_missing_or_blank_rows": int(data.attrs.get("filtered_missing_or_blank_rows", 0)),
        "duplicate_rows_removed": int(data.attrs.get("duplicate_rows_removed", 0)),
        "deduplication": "Trim, collapse whitespace, casefold; conflicting labels rejected",
        "dataset_rows_after_cleaning_and_deduplication": len(data),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "split_seed": seed,
        "test_size_fraction": test_size,
        "cross_validation_folds": folds,
        "note": "New comment-level baseline; not a reproduction of dissertation metrics or a video-held-out evaluation.",
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
