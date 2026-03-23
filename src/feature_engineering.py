import pandas as pd
from sklearn.preprocessing import StandardScaler


def feature_engineering(df):

    df = df.copy()

    # 1 Convert Boolean / Yes-No to Numeric

    if df["International plan"].dtype == "object":
        df["International plan"] = df["International plan"].map({"Yes":1,"No":0})

    if df["Voice mail plan"].dtype == "object":
        df["Voice mail plan"] = df["Voice mail plan"].map({"Yes":1,"No":0})

    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes":1,"No":0})


    # 2 Drop High Cardinality Column

    if "State" in df.columns:
        df = df.drop("State", axis=1)


    # 3 Remove Highly Correlated Features
    # Charges are derived from minutes

    correlated_features = [
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge"
    ]

    for col in correlated_features:
        if col in df.columns:
            df = df.drop(col, axis=1)


    # 4 Create New Useful Features

    df["Total minutes"] = (
        df["Total day minutes"]
        + df["Total eve minutes"]
        + df["Total night minutes"]
        + df["Total intl minutes"]
    )

    df["Total calls"] = (
        df["Total day calls"]
        + df["Total eve calls"]
        + df["Total night calls"]
        + df["Total intl calls"]
    )


    # 5 Feature Scaling

    scaler = StandardScaler()

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

    numeric_cols = numeric_cols.drop("Churn")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


    return df