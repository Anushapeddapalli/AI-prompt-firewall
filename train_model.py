
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from text_normalizer import normalize_batch


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "prompt_injection_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

# Save separately to protect the existing model.
MODEL_PATH = MODEL_DIR / "prompt_classifier_normalized.joblib"

MIN_VALIDATION_PRECISION = 0.85


def make_pipeline():
    return Pipeline([
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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {"prompt", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required}"
        )

    df = df.dropna(subset=["prompt", "label", "split"]).copy()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df = df[df["prompt"] != ""]

    label_map = {"benign": 0, "malicious": 1}
    unknown_labels = set(df["label"].unique()) - set(label_map)

    if unknown_labels:
        raise ValueError(f"Unexpected labels: {unknown_labels}")

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Both official train and test splits are required."
        )

    # Keep the dataset's official split intact.
    X_train_all = train_df["prompt"].reset_index(drop=True)
    y_train_all = train_df["label"].map(label_map).reset_index(drop=True)

    X_test_raw = test_df["prompt"].reset_index(drop=True)
    y_test = test_df["label"].map(label_map).reset_index(drop=True)

    # Normalize all prompts with the same function used at inference.
    print("Normalizing training prompts...")
    X_train_normalized = normalize_batch(X_train_all.tolist())

    print("Normalizing official test prompts...")
    X_test_normalized = normalize_batch(X_test_raw.tolist())

    # Validation data comes only from the official training split.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_normalized,
        y_train_all,
        test_size=0.20,
        random_state=42,
        stratify=y_train_all,
    )

    print(f"\nTraining subset: {len(X_fit)}")
    print(f"Validation subset: {len(X_val)}")
    print(f"Official test set: {len(X_test_normalized)}")

    # Select the decision threshold using validation data only.
    print("\nTraining temporary validation model...")
    validation_model = make_pipeline()
    validation_model.fit(X_fit, y_fit)

    val_scores = validation_model.predict_proba(X_val)[:, 1]
    candidates = []

    for threshold in np.arange(0.05, 0.951, 0.01):
        val_pred = (val_scores >= threshold).astype(int)

        precision = precision_score(
            y_val, val_pred, zero_division=0
        )
        recall = recall_score(
            y_val, val_pred, zero_division=0
        )
        f1 = f1_score(
            y_val, val_pred, zero_division=0
        )

        if precision >= MIN_VALIDATION_PRECISION:
            candidates.append(
                (recall, precision, f1, float(threshold))
            )

    if candidates:
        recall, precision, f1, threshold = max(candidates)

        print("\nThreshold selected on validation data:")
        print(f"Threshold: {threshold:.2f}")
        print(f"Validation precision: {precision:.4f}")
        print(f"Validation recall:    {recall:.4f}")
        print(f"Validation F1:        {f1:.4f}")
    else:
        threshold = 0.50
        print(
            "\nNo threshold met the validation precision target. "
            "Using 0.50 as a baseline."
        )

    # Train the final model on the complete official training split.
    print("\nTraining final model on official training data...")
    final_model = make_pipeline()
    final_model.fit(X_train_normalized, y_train_all)

    # Evaluate using the official test split.
    # Note: this test set has already been examined during development,
    # so these are development results, not a fresh blind evaluation.
    test_scores = final_model.predict_proba(
        X_test_normalized
    )[:, 1]

    predictions = (test_scores >= threshold).astype(int)

    print("\n===== NORMALIZED MODEL TEST RESULTS =====")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        "Malicious precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        "Malicious recall:    "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        "Malicious F1:        "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted)")
    print("Label order: benign, malicious")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": final_model,
            "threshold": float(threshold),
            "normalization": "text_normalizer.normalize_batch",
            "model_version": "normalized-v1",
        },
        MODEL_PATH,
    )

    print(f"\nSaved new normalized model to:\n{MODEL_PATH}")
    print("The previous prompt_classifier.joblib was not overwritten.")


if __name__ == "__main__":
    main()
