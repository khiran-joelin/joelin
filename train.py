#!/usr/bin/env python3
"""
Simple training script (RandomForest fallback)

Usage:
    python train_simple.py

Requirements:
    pip install pandas numpy scikit-learn joblib matplotlib seaborn
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Config (edit these if you want)
# ---------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = Path("./model_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = OUTPUT_DIR / "rf_model.joblib"
FEATURES_OUT = OUTPUT_DIR / "feature_importances.csv"
CONFUSION_PNG = OUTPUT_DIR / "confusion_matrix.png"
REPORT_TXT = OUTPUT_DIR / "classification_report.txt"

# Candidate filenames and directories to search
CANDIDATE_FULLS = [
    "fully_preprocessed_dataset_with_target_encoded.csv",
    "fully_cleaned_encoded_dataset.csv",
    "fully_preprocessed_dataset.csv",
    "fully_preprocessed.csv",
]
CANDIDATE_SPLIT = [
    ("X_train_preprocessed.csv", "y_train_preprocessed.csv", "X_test_preprocessed.csv", "y_test_preprocessed.csv"),
    ("X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"),
]

SEARCH_DIRS = [
    Path.cwd(),
    Path(__file__).resolve().parent,
    Path(r"D:\joelin\infosys springboard"),
    Path("/mnt/data"),
]

# ---------------------------
# Helpers
# ---------------------------
def find_first_file(names):
    """Search SEARCH_DIRS for any filename in names; return Path or None."""
    tried = []
    for d in SEARCH_DIRS:
        for n in names:
            p = (d / n)
            tried.append(str(p))
            if p.exists():
                return p
    return None

def find_split_files(tuple_names):
    """Try to find all four split files for each candidate tuple; returns tuple of Paths or None."""
    for names in tuple_names:
        paths = []
        found_all = True
        for n in names:
            p = find_first_file([n])
            if p is None:
                found_all = False
                break
            paths.append(p)
        if found_all:
            return tuple(paths)
    return None

def load_splits_if_present():
    sp = find_split_files(CANDIDATE_SPLIT)
    if sp is not None:
        X_train = pd.read_csv(sp[0])
        y_train = pd.read_csv(sp[1])
        X_test  = pd.read_csv(sp[2])
        y_test  = pd.read_csv(sp[3])
        # flatten y if single-col DataFrame
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
            y_test = y_test.iloc[:, 0]
        print(f"Loaded pre-split files: {sp}")
        return X_train, X_test, y_train, y_test
    return None

def load_full_and_split():
    p = find_first_file(CANDIDATE_FULLS)
    if p is None:
        # last resort: try any CSV in SEARCH_DIRS
        for d in SEARCH_DIRS:
            all_csvs = list(d.glob("*.csv"))
            if all_csvs:
                p = all_csvs[0]
                print(f"No canonical full file found; falling back to {p}")
                break
    if p is None:
        raise FileNotFoundError("Could not find any preprocessed CSV. Checked: " + ", ".join(str(d) for d in SEARCH_DIRS))
    print(f"Loading full dataset from: {p}")
    df = pd.read_csv(p)
    # find target column
    target_candidates = ["JobRole", "jobrole", "target", "label", "Job_Role"]
    target_col = None
    for c in target_candidates:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]
        print(f"Using last column '{target_col}' as target.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # convert float labels that are integers into ints
    if pd.api.types.is_float_dtype(y) and np.all(np.mod(y.dropna(), 1) == 0):
        y = y.astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def plot_and_save_confusion_matrix(y_true, y_pred, labels, out_fp):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_fp)
    plt.close()
    print(f"Saved confusion matrix to {out_fp}")

# ---------------------------
# Main
# ---------------------------
def main():
    splits = load_splits_if_present()
    if splits is not None:
        X_train, X_test, y_train, y_test = splits
    else:
        X_train, X_test, y_train, y_test = load_full_and_split()

    print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    # keep numeric columns only (assuming preprocessed)
    X_train = X_train.select_dtypes(include=[np.number]).copy()
    X_test  = X_test.select_dtypes(include=[np.number]).copy()

    # align columns
    for c in set(X_train.columns) - set(X_test.columns):
        X_test[c] = 0
    X_test = X_test[X_train.columns]

    # Train a simple RandomForest
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    print("Training RandomForest...")
    clf.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(clf_report)

    # Save classification report
    with open(REPORT_TXT, "w") as f:
        f.write(f"Test Accuracy: {acc:.6f}\n\n")
        f.write(clf_report)
    print(f"Saved classification report to {REPORT_TXT}")

    # Confusion matrix
    labels = sorted(list(unique_labels(y_test, y_pred)))
    plot_and_save_confusion_matrix(y_test, y_pred, labels, CONFUSION_PNG)

    # Feature importances
    try:
        fi_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(FEATURES_OUT, index=False)
        print(f"Saved feature importances to {FEATURES_OUT}")
    except Exception as e:
        print("Could not save feature importances:", e)

    # Save model
    joblib.dump(clf, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
