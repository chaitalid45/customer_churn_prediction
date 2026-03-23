# MODEL DEVELOPMENT

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering



# LOAD DATA
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


# MAIN FUNCTION
def train_model(train_path, test_path):

    # Load data
    train_df, test_df = load_data(train_path, test_path)

    # Preprocess
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Feature Engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Split X & y
    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]

    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # ALIGN COLUMNS (IMPORTANT)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


    #  SCALING (FIT ONLY ON TRAIN)
    scaler = StandardScaler()

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # SMOTE (ONLY TRAIN)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print("Before SMOTE:\n", y_train.value_counts())
    print("After SMOTE:\n", pd.Series(y_train_res).value_counts())

    # MODEL TRAINING
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    # FEATURE IMPORTANCE
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    print("\nTop Features:\n")
    print(feat_imp.sort_values(ascending=False).head(10))

    # PREDICTION
    y_prob = model.predict_proba(X_test)[:, 1]


    # THRESHOLD TUNING
    threshold = 0.4
    y_pred = (y_prob > threshold).astype(int)

    # METRICS
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model



if __name__ == "__main__":
    model = train_model(
        "../data/processed_data/train_processed.csv",
        "../data/processed_data/test_processed.csv"
    )