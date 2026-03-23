import pandas as pd
import numpy as np


def missing_values(df):
    df = df.copy()

    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif df[col].dtype == 'bool':
                continue
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


def find_outliers(df):
    df = df.copy()

    for col in df.columns:
        if col == "Churn":
            continue

        if df[col].dtype in ["object", "bool"]:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Apply log only if positive values
        if (df[col] > 0).all():
            if ((df[col] < lower) | (df[col] > upper)).sum() > 0:
                df[col] = np.log1p(df[col])

    return df


def preprocess_data(df):
    df = missing_values(df)
    df = find_outliers(df)
    df = df.drop_duplicates()

    return df