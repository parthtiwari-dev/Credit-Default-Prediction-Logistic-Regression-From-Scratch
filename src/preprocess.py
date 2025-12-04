# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json

RANDOM_STATE = 42
RAW_PATH = os.path.join("data", "raw.csv")
PROCESSED_PATH = os.path.join("data", "processed.csv")
SCALER_PATH = os.path.join("outputs", "scaler.joblib")
META_PATH = os.path.join("outputs", "meta.json")

def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, index_col=None)
    return df

def clean_columns(df):
    # Keep original names but make them consistent
    df.columns = [c.strip() for c in df.columns]
    # The UCI dataset has an initial 'ID' column usually named 'ID' or 'ID#'
    return df

def fix_education(df):
    # Education has values 1,2,3,4, and other codes (0,5,6) -> map to 'Other'
    df['EDUCATION'] = df['EDUCATION'].replace({0:4, 5:4, 6:4})
    return df

def fix_marriage(df):
    df['MARRIAGE'] = df['MARRIAGE'].replace({0:3})  # 3 -> Unknown/Other
    return df

def create_age_bins(df):
    bins = [18, 30, 40, 50, 60, 100]
    labels = ['18-29', '30-39', '40-49', '50-59', '60+']
    df['AGE_BIN'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)
    return df

def one_hot_encode(df, cols):
    return pd.get_dummies(df, columns=cols, drop_first=True)

def scale_numeric(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump(scaler, SCALER_PATH)
    return df, scaler

def preprocess(save=True):
    df = load_raw()
    df = clean_columns(df)
    # If raw file uses chinese header row, make sure target exists
    if 'default.payment.next.month' not in df.columns:
        raise KeyError("Target 'default.payment.next.month' not found in raw.csv")
    df = fix_education(df)
    df = fix_marriage(df)
    df = create_age_bins(df)
    # selected features to keep (recommended)
    keep = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
        'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
        'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
        'AGE_BIN',
        'default.payment.next.month'
    ]
    # If dataset has slightly different names, select intersection
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    # Fix PAY_0 column name if present as PAY_1 in some mirrors
    if 'PAY_1' in df.columns and 'PAY_0' not in df.columns:
        df.rename(columns={'PAY_1':'PAY_0'}, inplace=True)
    # Clean payment status: ensure numeric
    pay_cols = [c for c in df.columns if c.startswith('PAY_')]
    df[pay_cols] = df[pay_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    # One-hot encode categorical: SEX, EDUCATION, MARRIAGE, AGE_BIN
    cat_cols = [c for c in ['SEX','EDUCATION','MARRIAGE','AGE_BIN'] if c in df.columns]
    df = one_hot_encode(df, cat_cols)
    # Numeric columns scaling
    numeric_cols = [c for c in df.columns if c not in ['default.payment.next.month'] and df[c].dtype in [np.float64, np.int64]]
    # exclude one-hot dummies (they are numeric but small). We'll scale only continuous variables
    continuous = [c for c in ['LIMIT_BAL','AGE'] if c in numeric_cols] + \
                 [c for c in df.columns if c.startswith('BILL_AMT') or c.startswith('PAY_AMT')]
    continuous = [c for c in continuous if c in df.columns]
    df, scaler = scale_numeric(df, continuous)
    # Save
    
    if save:
        df.to_csv(PROCESSED_PATH, index=False)
        meta = {
            "continuous": continuous,
            "all_columns": df.columns.tolist(),
            "target": "default.payment.next.month",
        }
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
    return df

if __name__ == "__main__":
    df = preprocess()
    print("Processed saved to data/processed.csv; shape:", df.shape)
