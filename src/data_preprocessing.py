### data preprocessing

import pandas as pd
import numpy as np

## Missing values and imputing of the values

def missing_values(df):
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace = True)
            elif df[col].dtype == 'bool':
                continue
            else:
                df[col].fillna(df[col].mean(), inplace = True)
    return df


## Outliers detection and log transformation

def find_outliers(df):
    for col in df.columns:
        if col == "Churn":
            continue

        d = df[col]
        if d.dtype == "bool" or d.dtype == "object":
            continue
        q1 = np.quantile(d,0.25)
        q3 = np.quantile(d, 0.75)
        IQR = q3 - q1
        lower = q1 - 1.5 * IQR
        upper = q3 + 1.5 * IQR
        if (d < lower).any() or (d > upper).any():
            df[col] = np.log1p(df[col])

    return df

def preprocessed_data(df):
    df = missing_values(df)
    df = find_outliers(df)
    df = df.drop_duplicated()
    return df