# retrain_rf_balanced.py
import os
from pathlib import Path
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ---------- CONFIG ----------
# 👉 CHANGE THESE TO YOUR REAL FILE PATHS
DATA_DIR = Path(r"D:\joelin\infosys springboard")

X_FP = DATA_DIR / "X_train_preprocessed.csv"
y_FP = DATA_DIR / "y_train_preprocessed.csv"

OUT_MODEL = Path("model_output") / "rf_model.joblib"
OUT_META = Path("models") / "model_metadata.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------- load files ----------
if not X_FP.exists() or not y_FP.exists():
    raise FileNotFoundError(f"Missing files:\n{X_FP}\n{y_FP}")

print("Loading:", X_FP)
X = pd.read_csv(X_FP)

print("Loading:", y_FP)
y = pd.read_csv(y_FP)

if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
    y = y.iloc[:, 0]

# ---------- splits ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print("Training shape:", X_train.shape)

# ---------- train balanced RF ----------
clf = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
)
clf.fit(X_train, y_train)

# ---------- evaluation ----------
y_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, zero_division=0))

# ---------- save ----------
OUT_MODEL.parent.mkdir(exist_ok=True)
joblib.dump(clf, OUT_MODEL)
print("Saved model to", OUT_MODEL)

meta = {
    "feature_order": list(X.columns),
    "classes": clf.classes_.tolist()
}
OUT_META.parent.mkdir(exist_ok=True)
with open(OUT_META, "wb") as f:
    pickle.dump(meta, f)

print("Saved metadata to", OUT_META)
print("DONE")
