
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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
REPORT_DIR = BASE_DIR / "reports"
MIN_VALIDATION_PRECISION = 0.85


def make_pipeline(use_normalizer=False):
    steps = []

    if use_normalizer:
        steps.append((
            "normalize",
            FunctionTransformer(normalize_batch, validate=False),
        ))

    steps.extend([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=100_000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
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
            candidates.append((recall, precision, f1, threshold))

    if not candidates:
        return 0.50

    return float(max(candidates)[3])


def main():
    df = pd.read_csv(DATA_PATH)

    required = {"prompt", "label", "split", "category", "technique"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    df = df.dropna(subset=["prompt", "label", "split"]).copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].astype(str).str.lower().str.strip()
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

    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Official test examples: {len(X_test)}")

    all_metrics = []
    prediction_frames = []

    for model_name, use_normalizer in [
        ("Baseline", False),
        ("Normalized", True),
    ]:
        print(f"\n{'=' * 12} {model_name} {'=' * 12}")

        # Select threshold using training-derived validation data only.
        validation_model = make_pipeline(use_normalizer)
        validation_model.fit(X_fit, y_fit)

        validation_scores = validation_model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, validation_scores)

        print(f"Selected threshold: {threshold:.2f}")

        # Fit on all official training examples; never fit on test rows.
        model = make_pipeline(use_normalizer)
        model.fit(X_train, y_train)

        scores = model.predict_proba(X_test)[:, 1]
        predictions = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        benign_count = tn + fp
        malicious_count = tp + fn

        metrics = {
            "model": model_name,
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "false_positive_rate": (
                fp / benign_count if benign_count else 0.0
            ),
            "false_negative_rate": (
                fn / malicious_count if malicious_count else 0.0
            ),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }
        all_metrics.append(metrics)

        print(f"Accuracy:            {metrics['accuracy']:.4f}")
        print(f"Malicious precision:  {metrics['precision']:.4f}")
        print(f"Malicious recall:     {metrics['recall']:.4f}")
        print(f"Malicious F1:         {metrics['f1']:.4f}")
        print(f"False-positive rate:  {metrics['false_positive_rate']:.4f}")
        print(f"False-negative rate:  {metrics['false_negative_rate']:.4f}")
        print(f"Confusion matrix [[TN, FP], [FN, TP]]: [[{tn}, {fp}], [{fn}, {tp}]]")

        result = test_df[[
            "id", "prompt", "label", "category", "technique"
        ]].copy()
        result["actual"] = y_test.to_numpy()
        result["predicted"] = predictions
        result["risk_score"] = scores
        result["threshold"] = threshold
        result["model"] = model_name
        result["error"] = result["actual"] != result["predicted"]
        prediction_frames.append(result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = REPORT_DIR / "final_ml_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = REPORT_DIR / "final_ml_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # Evaluate recall only on categories that contain malicious examples.
    malicious_test = test_df[test_df["label"] == "malicious"]
    category_rows = []

    for category, group in malicious_test.groupby("category"):
        row = {
            "category": category,
            "malicious_examples": len(group),
        }

        for model_name in ["Baseline", "Normalized"]:
            model_results = predictions_df[
                (predictions_df["model"] == model_name)
                & (predictions_df["category"] == category)
                & (predictions_df["actual"] == 1)
            ]
            row[f"{model_name.lower()}_recall"] = (
                model_results["predicted"].mean()
            )

        category_rows.append(row)

    category_df = pd.DataFrame(category_rows).sort_values("category")
    category_path = REPORT_DIR / "malicious_category_recall.csv"
    category_df.to_csv(category_path, index=False)

    print("\n========== SIDE-BY-SIDE METRICS ==========")
    print(metrics_df.round(4).to_string(index=False))

    print("\n========== MALICIOUS CATEGORY RECALL ==========")
    print(category_df.round(4).to_string(index=False))

    # Show up to five missed malicious prompts for each model.
    print("\n========== MISSED MALICIOUS EXAMPLES ==========")
    for model_name in ["Baseline", "Normalized"]:
        missed = predictions_df[
            (predictions_df["model"] == model_name)
            & (predictions_df["actual"] == 1)
            & (predictions_df["predicted"] == 0)
        ]

        print(f"\n{model_name}: {len(missed)} missed attacks")
        for _, row in missed.head(5).iterrows():
            print(f"- [{row['category']} / {row['technique']}] "
                  f"score={row['risk_score']:.4f}: {row['prompt'][:240]}")

    print("\nReports saved:")
    print(metrics_path)
    print(predictions_path)
    print(category_path)
    print("\nExisting saved model was not modified.")


if __name__ == "__main__":
    main()
