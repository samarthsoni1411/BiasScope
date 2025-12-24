# modules/data_utils.py
import os, uuid
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PROC_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file) -> str:
    uid = uuid.uuid4().hex[:8]
    filename = f"{uid}_{uploaded_file.name}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def read_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type")

def guess_target_column(df: pd.DataFrame) -> Optional[str]:
    common = ["target", "label", "class", "y", "outcome", "response", "income"]
    for c in df.columns:
        if c.lower() in common:
            return c
    for c in df.columns:
        try:
            if df[c].nunique() < 30 and not pd.api.types.is_float_dtype(df[c]):
                return c
        except Exception:
            continue
    return None

def preprocess_data(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[str, pd.DataFrame, list]:
    df_clean = df.copy()

    # Fill missing
    for c in df_clean.columns:
        if df_clean[c].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[c]):
                df_clean[c] = df_clean[c].fillna(df_clean[c].median())
            else:
                try:
                    df_clean[c] = df_clean[c].fillna(df_clean[c].mode().iloc[0])
                except Exception:
                    df_clean[c] = df_clean[c].fillna("NAN")

    # Encode categoricals (per-column LabelEncoder)
    encoded_cols = []
    for c in df_clean.columns:
        if c == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df_clean[c]):
            le = LabelEncoder()
            df_clean[c] = le.fit_transform(df_clean[c].astype(str))
            encoded_cols.append(c)

    # Scale numeric columns excluding target
    numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != target_col]
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    out_path = os.path.join(PROC_DIR, f"{uuid.uuid4().hex[:8]}_cleaned.csv")
    df_clean.to_csv(out_path, index=False)
    return out_path, df_clean, encoded_cols
