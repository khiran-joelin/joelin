# fit_and_save_encoders.py
import os
from pathlib import Path
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# CONFIG - adjust source path if needed
RAW_FP = Path("D:\infosys\cleaned_dataset_before_encoding.csv")   # <-- change if necessary
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENC_CATEGORICAL_FP = Path("categorical_label_encoders.pkl")
ENC_NUMERIC_FP = Path("numerical_scaler.pkl")
ENC_TARGET_FP = Path("target_encoder.pkl")
META_FP = OUT_DIR / "model_metadata.pkl"
# Also write preprocessed CSVs for retraining convenience
X_PRE_FP = Path("/mnt/data/X_train_preprocessed.csv")
y_PRE_FP = Path("/mnt/data/y_train_preprocessed.csv")

if not RAW_FP.exists():
    raise FileNotFoundError(f"Could not find raw cleaned dataset at {RAW_FP}. Update RAW_FP in this script.")

print("Loading dataset:", RAW_FP)
df = pd.read_csv(RAW_FP)

# Identify target column
target_candidates = ["JobRole","Job_Role","Job Role","jobrole","job_role","target","label"]
target_col = None
for c in target_candidates:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    # fallback: last column
    target_col = df.columns[-1]
    print(f"Warning: couldn't find known target names. Using last column as target: {target_col}")

print("Using target column:", target_col)

# Drop rows with missing target
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# Split X/y
X = df.drop(columns=[target_col])
y = df[target_col].astype(str).reset_index(drop=True)

# Detect categorical vs numeric
cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype == "bool" or X[c].nunique() < 50]
# remove any obviously numeric columns from cat detection (if pandas mis-typed)
# ensure we don't treat small-int numeric as categorical if it's real numeric
num_cols = [c for c in X.columns if c not in cat_cols]

# sometimes columns like 'Age' are int but counted as categorical because small nunique; fix heuristics:
# if column is numeric dtype, treat as numeric
for c in X.columns:
    if pd.api.types.is_numeric_dtype(X[c]):
        if c in cat_cols:
            cat_cols.remove(c)
        if c not in num_cols:
            num_cols.append(c)

print("Categorical columns detected:", cat_cols)
print("Numeric columns detected:", num_cols)

# Prepare encoders dictionary
categorical_encoders = {}
# Use OrdinalEncoder for each categorical column separately to preserve mapping & allow inverse_transform if needed
for c in cat_cols:
    vals = X[c].fillna("##MISSING##").astype(str).values.reshape(-1, 1)
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    oe.fit(vals)
    categorical_encoders[c] = oe
    # transform and replace in X for the preprocessed CSV
    X[c] = oe.transform(vals).astype(float).ravel()

# Numeric scaler for numeric columns (only if any numeric cols)
numerical_scaler = None
if num_cols:
    scaler = StandardScaler()
    # fillna with median before scaling to avoid errors
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    scaler.fit(X[num_cols].values)
    X[num_cols] = scaler.transform(X[num_cols].values)
    numerical_scaler = scaler

# Target encoder
target_encoder = LabelEncoder()
y_enc = target_encoder.fit_transform(y.values)

# Save preprocessed CSVs for quick retrain
X_PRE_FP.parent.mkdir(parents=True, exist_ok=True)
X.to_csv(X_PRE_FP, index=False)
pd.DataFrame(y_enc, columns=["target"]).to_csv(y_PRE_FP, index=False)
print("Saved preprocessed X to:", X_PRE_FP)
print("Saved preprocessed y to:", y_PRE_FP)

# Save encoders
joblib.dump(categorical_encoders, ENC_CATEGORICAL_FP)
print("Saved categorical encoders to:", ENC_CATEGORICAL_FP)
if numerical_scaler is not None:
    joblib.dump(numerical_scaler, ENC_NUMERIC_FP)
    print("Saved numeric scaler to:", ENC_NUMERIC_FP)
else:
    print("No numeric scaler saved (no numeric cols).")
joblib.dump(target_encoder, ENC_TARGET_FP)
print("Saved target encoder to:", ENC_TARGET_FP)

# Save metadata
meta = {
    "feature_order": list(X.columns),
    "classes": target_encoder.classes_.tolist(),
    "categorical_columns": cat_cols,
    "numeric_columns": num_cols
}
with open(META_FP, "wb") as f:
    pickle.dump(meta, f)
print("Saved metadata to:", META_FP)

print("Done. You can now retrain the model using X_train_preprocessed.csv and y_train_preprocessed.csv or restart the app to use encoders.")
