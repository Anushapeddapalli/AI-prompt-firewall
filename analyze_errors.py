
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "prompt_injection_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "prompt_classifier.joblib"
REPORT_DIR = BASE_DIR / "reports"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["prompt", "label", "split"]).copy()

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["prompt"] = df["prompt"].astype(str)

    test = df[df["split"] == "test"].copy()
    test = test[test["label"].isin(["benign", "malicious"])]

    saved = joblib.load(MODEL_PATH)
    model = saved["pipeline"]
    threshold = float(saved["threshold"])

    scores = model.predict_proba(test["prompt"])[:, list(model.classes_).index(1)]
    test["risk_score"] = scores
    test["predicted"] = [
        "malicious" if score >= threshold else "benign"
        for score in scores
    ]

    test["error_type"] = "correct"
    test.loc[
        (test["label"] == "malicious") &
        (test["predicted"] == "benign"),
        "error_type",
    ] = "false_negative"

    test.loc[
        (test["label"] == "benign") &
        (test["predicted"] == "malicious"),
        "error_type",
    ] = "false_positive"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep prompt text in the local report for manual error analysis.
    test.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    print("Errors by dataset category:")
    print(
        test[test["error_type"] != "correct"]
        .groupby(["category", "error_type"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nErrors by attack technique:")
    print(
        test[test["label"] == "malicious"]
        .groupby("technique")
        .apply(
            lambda group: pd.Series({
                "examples": len(group),
                "missed": int((group["predicted"] == "benign").sum()),
                "miss_rate": round(
                    (group["predicted"] == "benign").mean(), 3
                ),
            }),
            include_groups=False,
        )
        .sort_values("miss_rate", ascending=False)
        .to_string()
    )

    print("\nFive missed malicious examples:")
    missed = test[
        (test["label"] == "malicious") &
        (test["predicted"] == "benign")
    ].sort_values("risk_score")

    for _, row in missed.head(5).iterrows():
        print(f"\nCategory: {row.get('category', 'unknown')}")
        print(f"Technique: {row.get('technique', 'unknown')}")
        print(f"Risk score: {row['risk_score']:.4f}")
        print(f"Prompt: {row['prompt'][:500]}")

    print("\nConfusion matrix (actual rows: benign, malicious):")
    print(confusion_matrix(
        test["label"],
        test["predicted"],
        labels=["benign", "malicious"],
    ))

    print(f"\nFull local report saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
