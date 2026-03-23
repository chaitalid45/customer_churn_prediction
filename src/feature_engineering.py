import pandas as pd


def feature_engineering(df):
    df = df.copy()

    mapping = {"Yes": 1, "No": 0}

    for col in ["International plan", "Voice mail plan", "Churn"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map(mapping)

    if "State" in df.columns:
        df = df.drop("State", axis=1)


    correlated_features = [
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge"
    ]

    df = df.drop(columns=[col for col in correlated_features if col in df.columns], errors='ignore')

    required_min_cols = [
        "Total day minutes", "Total eve minutes",
        "Total night minutes", "Total intl minutes"
    ]

    if all(col in df.columns for col in required_min_cols):
        df["Total minutes"] = df[required_min_cols].sum(axis=1)

    required_call_cols = [
        "Total day calls", "Total eve calls",
        "Total night calls", "Total intl calls"
    ]

    if all(col in df.columns for col in required_call_cols):
        df["Total calls"] = df[required_call_cols].sum(axis=1)

    return df