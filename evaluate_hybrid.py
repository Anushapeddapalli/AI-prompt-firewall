
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from predict import analyze_prompt


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "prompt_injection_dataset.csv"
REPORT_DIR = BASE_DIR / "reports"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"prompt", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required}"
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

    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    if test_df.empty:
        raise ValueError("No official test examples were found.")

    actual = test_df["label"].map(label_map).tolist()
    results = []

    print(f"Evaluating hybrid detector on {len(test_df)} prompts...")

    for index, prompt in enumerate(test_df["prompt"], start=1):
        result = analyze_prompt(prompt)
        results.append(result)

        if index % 200 == 0 or index == len(test_df):
            print(f"Processed {index}/{len(test_df)} prompts")

    report = test_df[["prompt", "label"]].copy()
    report["actual"] = actual
    report["predicted"] = [
        1 if result["label"] == "suspicious" else 0
        for result in results
    ]
    report["risk_score"] = [
        result["risk_score"] for result in results
    ]
    report["threshold"] = [
        result.get("threshold") for result in results
    ]
    report["detection_path"] = [
        result["detection_path"] for result in results
    ]
    report["reason"] = [
        result["reason"] for result in results
    ]
    report["correct"] = report["actual"] == report["predicted"]

    y_true = report["actual"]
    y_pred = report["predicted"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    print("\n===== HYBRID DETECTOR RESULTS =====")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Malicious precision:  {precision:.4f}")
    print(f"Malicious recall:     {recall:.4f}")
    print(f"Malicious F1:         {f1:.4f}")
    print(f"False-positive rate:  {false_positive_rate:.4f}")
    print(f"False-negative rate:  {false_negative_rate:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            zero_division=0,
        )
    )

    print("\n===== DETECTION PATH COUNTS =====")
    print(report["detection_path"].value_counts(dropna=False).to_string())

    print("\n===== DETECTION PATH BY ACTUAL LABEL =====")
    print(
        pd.crosstab(
            report["label"],
            report["detection_path"],
            margins=True,
        ).to_string()
    )

    rule_rows = report[
        report["detection_path"] == "decoded_override_rule"
    ]

    print("\n===== RULE-TRIGGERED CASES =====")
    print(f"Rule-triggered prompts: {len(rule_rows)}")

    if not rule_rows.empty:
        print(
            rule_rows[
                ["label", "predicted", "risk_score", "prompt"]
            ].head(10).to_string(index=False)
        )
    else:
        print(
            "No official test prompts triggered the decoded override rule."
        )

    errors = report[~report["correct"]]

    print("\n===== FIRST 10 INCORRECT PREDICTIONS =====")
    if errors.empty:
        print("No incorrect predictions.")
    else:
        print(
            errors[
                [
                    "label",
                    "predicted",
                    "risk_score",
                    "detection_path",
                    "prompt",
                ]
            ].head(10).to_string(index=False)
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = REPORT_DIR / "hybrid_test_predictions.csv"
    errors_path = REPORT_DIR / "hybrid_test_errors.csv"
    metrics_path = REPORT_DIR / "hybrid_test_metrics.csv"

    report.to_csv(predictions_path, index=False)
    errors.to_csv(errors_path, index=False)

    metrics = pd.DataFrame([{
        "model": "hybrid",
        "test_examples": len(report),
        "threshold": float(results[0]["threshold"])
        if results and results[0].get("threshold") is not None
        else None,
        "accuracy": accuracy,
        "malicious_precision": precision,
        "malicious_recall": recall,
        "malicious_f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "rule_triggered_count": len(rule_rows),
    }])
    metrics.to_csv(metrics_path, index=False)

    print("\nSaved reports:")
    print(predictions_path)
    print(errors_path)
    print(metrics_path)

    print(
        "\nCaution: this official test split has already been inspected "
        "during development. These are development metrics, not an "
        "independent security benchmark."
    )


if __name__ == "__main__":
    main()
