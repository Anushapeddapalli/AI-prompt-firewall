
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from text_normalizer import normalize_batch


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "prompt_injection_dataset.csv"

MIN_VALIDATION_PRECISION = 0.85


def make_pipeline(use_normalizer=False):
    steps = []

    if use_normalizer:
        steps.append(
            (
                "normalize",
                FunctionTransformer(
                    normalize_batch,
                    validate=False,
                ),
            )
        )

    steps.extend([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=100_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    return Pipeline(steps)


def select_threshold(y_true, scores):
    candidates = []

    for threshold in np.arange(0.05, 0.951, 0.01):
        predictions = (scores >= threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        if precision >= MIN_VALIDATION_PRECISION:
            candidates.append(
                (recall, precision, f1, threshold)
            )

    if not candidates:
        return 0.50

    return float(max(candidates)[3])


def evaluate(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    required = {"prompt", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    df = df.dropna(subset=["prompt", "label", "split"]).copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df = df[df["prompt"] != ""]

    label_map = {"benign": 0, "malicious": 1}
    if not set(df["label"].unique()).issubset(label_map):
        raise ValueError("Unexpected labels in dataset.")

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Official train and test splits are required.")

    X_train = train_df["prompt"]
    y_train = train_df["label"].map(label_map)
    X_test = test_df["prompt"]
    y_test = test_df["label"].map(label_map)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    results = []

    for name, use_normalizer in [
        ("Baseline", False),
        ("Normalized", True),
    ]:
        print(f"\n{'=' * 16} {name} {'=' * 16}")

        validation_model = make_pipeline(use_normalizer)
        validation_model.fit(X_fit, y_fit)

        val_scores = validation_model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_scores)

        val_metrics = evaluate(y_val, val_scores, threshold)

        print(f"Validation threshold: {threshold:.2f}")
        print("Validation metrics:", {
            key: round(value, 4)
            for key, value in val_metrics.items()
        })

        # Refit on the complete official training split.
        final_model = make_pipeline(use_normalizer)
        final_model.fit(X_train, y_train)

        test_scores = final_model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, test_scores, threshold)

        print("Official test metrics:", {
            key: round(value, 4)
            for key, value in test_metrics.items()
        })

        results.append({
            "model": name,
            "threshold": threshold,
            **{
                f"validation_{key}": value
                for key, value in val_metrics.items()
            },
            **{
                f"test_{key}": value
                for key, value in test_metrics.items()
            },
        })

    comparison = pd.DataFrame(results).set_index("model")

    print("\n\n========== SIDE-BY-SIDE COMPARISON ==========")
    print(comparison.round(4).to_string())

    report_dir = BASE_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "normalization_comparison.csv"
    comparison.to_csv(report_path)

    print(f"\nComparison saved to: {report_path}")
    print("\nNo existing model file was overwritten.")


if __name__ == "__main__":
    main()
