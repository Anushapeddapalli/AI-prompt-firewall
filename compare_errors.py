
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from text_normalizer import normalize_batch


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "prompt_injection_dataset.csv"


def make_pipeline(normalized):
    steps = []

    if normalized:
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


def choose_threshold(y, scores):
    candidates = []

    for threshold in [i / 100 for i in range(5, 96)]:
        pred = (scores >= threshold).astype(int)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)

        if precision >= 0.85:
            candidates.append((recall, precision, f1, threshold))

    return max(candidates)[3] if candidates else 0.50


def main():
    df = pd.read_csv(DATA).dropna(
        subset=["prompt", "label", "split"]
    ).copy()

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"] != ""]

    label_map = {"benign": 0, "malicious": 1}
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].copy()

    X_train = train["prompt"]
    y_train = train["label"].map(label_map)
    X_test = test["prompt"]
    y_test = test["label"].map(label_map)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    summaries = []

    for name, normalized in [
        ("Baseline", False),
        ("Normalized", True),
    ]:
        model = make_pipeline(normalized)
        model.fit(X_fit, y_fit)

        val_scores = model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_scores)

        # Refit using only the official training split.
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        predictions = (scores >= threshold).astype(int)

        results = test.copy()
        results["actual"] = y_test.to_numpy()
        results[name + "_pred"] = predictions

        summaries.append(results[[
            "id", "category", "technique", "actual", name + "_pred"
        ]].set_index("id"))

        print(f"\n{name}: threshold={threshold:.2f}")

    merged = summaries[0].join(
        summaries[1][["Normalized_pred"]],
        how="inner",
    )

    rows = []
    for category, group in merged.groupby("category"):
        actual = group["actual"]
        row = {"category": category, "examples": len(group)}

        for name in ["Baseline", "Normalized"]:
            pred = group[name + "_pred"]
            row[f"{name}_recall"] = recall_score(
                actual, pred, zero_division=0
            )
            row[f"{name}_precision"] = precision_score(
                actual, pred, zero_division=0
            )
            row[f"{name}_f1"] = f1_score(
                actual, pred, zero_division=0
            )

        rows.append(row)

    report = pd.DataFrame(rows).sort_values("category")
    print("\nPerformance by attack category:")
    print(report.round(3).to_string(index=False))

    output = BASE / "reports" / "category_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(f"\nSaved report to: {output}")


if __name__ == "__main__":
    main()
