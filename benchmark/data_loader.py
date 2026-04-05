# benchmark/data_loader.py
"""
Downloads and prepares the 3 benchmark datasets for BiasScope publication evaluation.

Datasets:
  1. Adult Income (UCI) — gender/race bias in income prediction
  2. COMPAS Recidivism (ProPublica) — racial bias in criminal justice
  3. German Credit (UCI Statlog) — age/gender bias in credit scoring

All datasets are public domain / open-access.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Adult Income Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_adult_income() -> pd.DataFrame:
    """
    UCI Adult Income dataset (Kohavi, 1996).
    Task: Predict income > $50K/year.
    Sensitive attributes: sex, race
    ~48,842 rows.
    """
    cache_path = DATA_DIR / "adult_income.csv"
    if cache_path.exists():
        print(f"  [Cache] Loading Adult Income from {cache_path}")
        return pd.read_csv(cache_path)

    print("  [Download] Fetching Adult Income dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    try:
        df = pd.read_csv(url, names=cols, skipinitialspace=True, na_values="?")
    except Exception:
        # Fallback: download raw bytes
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), names=cols, skipinitialspace=True, na_values="?")

    df = df.dropna().reset_index(drop=True)

    # Standardise target
    df["income"] = (df["income"].str.strip() == ">50K").astype(int)

    # Keep relevant columns
    keep = ["age", "education_num", "occupation", "relationship", "race",
            "sex", "capital_gain", "capital_loss", "hours_per_week", "income"]
    df = df[keep]

    df.to_csv(cache_path, index=False)
    print(f"  [Saved] {len(df)} rows → {cache_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPAS Recidivism Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_compas() -> pd.DataFrame:
    """
    ProPublica COMPAS dataset (Angwin et al., 2016).
    Task: Predict two-year recidivism.
    Sensitive attributes: race, sex
    ~6,172 rows (filtered, following ProPublica methodology).
    """
    cache_path = DATA_DIR / "compas.csv"
    if cache_path.exists():
        print(f"  [Cache] Loading COMPAS from {cache_path}")
        return pd.read_csv(cache_path)

    print("  [Download] Fetching COMPAS dataset...")
    url = (
        "https://raw.githubusercontent.com/propublica/compas-analysis/"
        "master/compas-scores-two-years.csv"
    )
    try:
        df = pd.read_csv(url)
    except Exception:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

    # Follow ProPublica filtering methodology
    df = df[
        (df["days_b_screening_arrest"] <= 30) &
        (df["days_b_screening_arrest"] >= -30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O") &
        (df["score_text"] != "N/A")
    ].reset_index(drop=True)

    keep = ["age", "c_charge_degree", "race", "sex", "priors_count",
            "days_b_screening_arrest", "decile_score", "is_recid"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.dropna().reset_index(drop=True)
    df = df.rename(columns={"is_recid": "recidivism"})

    df.to_csv(cache_path, index=False)
    print(f"  [Saved] {len(df)} rows → {cache_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. German Credit Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_german_credit() -> pd.DataFrame:
    """
    UCI Statlog German Credit dataset (Hofmann, 1994).
    Task: Predict credit risk (good/bad).
    Sensitive attributes: age (binary: ≥25 = privileged), personal_status (sex proxy)
    1,000 rows.
    """
    cache_path = DATA_DIR / "german_credit.csv"
    if cache_path.exists():
        print(f"  [Cache] Loading German Credit from {cache_path}")
        return pd.read_csv(cache_path)

    print("  [Download] Fetching German Credit dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    cols = [
        "checking_account", "duration", "credit_history", "purpose",
        "credit_amount", "savings", "employment", "installment_rate",
        "personal_status", "other_debtors", "residence_since", "property",
        "age", "other_plans", "housing", "existing_credits", "job",
        "liable_people", "telephone", "foreign_worker", "credit_risk"
    ]
    try:
        df = pd.read_csv(url, names=cols, sep=" ", header=None)
    except Exception:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text), names=cols, sep=" ", header=None)

    # credit_risk: 1=good, 2=bad → recode to 0=good, 1=bad
    df["credit_risk"] = (df["credit_risk"] == 2).astype(int)

    # Create age_group sensitive attribute (standard fairness convention)
    df["age_group"] = (df["age"] >= 25).map({True: "old", False: "young"})

    # Extract sex from personal_status (A91=male, A92/A95=female, A93/A94=male)
    male_codes = {"A91", "A93", "A94"}
    df["sex"] = df["personal_status"].apply(
        lambda x: "Male" if x in male_codes else "Female"
    )

    keep = ["duration", "credit_amount", "installment_rate", "residence_since",
            "age", "age_group", "sex", "existing_credits", "credit_risk"]
    df = df[keep].dropna().reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    print(f"  [Saved] {len(df)} rows → {cache_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "adult_income": {
        "loader": load_adult_income,
        "target": "income",
        "sensitive_marginal": [["sex"], ["race"]],
        "sensitive_intersectional": ["sex", "race"],
        "privileged": {"sex": "Male", "race": "White"},
        "description": "UCI Adult Income — income > $50K prediction",
    },
    "compas": {
        "loader": load_compas,
        "target": "recidivism",
        "sensitive_marginal": [["race"], ["sex"]],
        "sensitive_intersectional": ["race", "sex"],
        "privileged": {"race": "Caucasian", "sex": "Male"},
        "description": "ProPublica COMPAS — two-year recidivism prediction",
    },
    "german_credit": {
        "loader": load_german_credit,
        "target": "credit_risk",
        "sensitive_marginal": [["age_group"], ["sex"]],
        "sensitive_intersectional": ["age_group", "sex"],
        "privileged": {"age_group": "old", "sex": "Male"},
        "description": "UCI German Credit — credit risk prediction",
    },
}


if __name__ == "__main__":
    print("=== Downloading/verifying all benchmark datasets ===\n")
    for name, cfg in DATASETS.items():
        print(f"\n[{name.upper()}] {cfg['description']}")
        df = cfg["loader"]()
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Target distribution:\n{df[cfg['target']].value_counts().to_string()}")
    print("\n✅ All datasets ready.")
