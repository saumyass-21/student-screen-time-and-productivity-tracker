
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    BASE_DIR, "output", "1_data_cleaning", "cleaned_dataset.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "4_linear_regression")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_model():
    print("\n========== LINEAR REGRESSION ==========")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "Cleaned dataset not found. Run data cleaning first."
        )

    df = pd.read_csv(DATA_FILE)

    target = "daily_screen_time_hours"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    # Separate features and target
    excluded_columns = [
        target,
        "addicted_label",
        "addiction_level"
    ]

    X = df.drop(
        columns=[
            col for col in excluded_columns
            if col in df.columns
        ]
    )

    y = df[target]

    # Identify numerical and categorical features
    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if X.shape[1] == 0:
        raise ValueError("No predictor features are available.")

    # Numerical preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Feature selection and regression pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(
            score_func=f_regression,
            k=10
        )),
        ("regression", LinearRegression())
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Input features:", X.shape[1])

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    # Save metrics
    metrics_text = f"""
LINEAR REGRESSION MODEL RESULTS

Target Variable: {target}

Training Records: {len(X_train)}
Testing Records: {len(X_test)}

Mean Absolute Error (MAE): {mae:.4f}
Mean Squared Error (MSE): {mse:.4f}
Root Mean Squared Error (RMSE): {rmse:.4f}
R2 Score: {r2:.4f}

5-Fold Cross Validation Scores:
{cv_scores}

Average Cross Validation R2:
{cv_scores.mean():.4f}
"""

    print(metrics_text)

    with open(
        os.path.join(OUTPUT_DIR, "model_metrics.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(metrics_text)

    # Save actual and predicted values
    predictions = pd.DataFrame({
        "Actual Screen Time": y_test.values,
        "Predicted Screen Time": y_pred,
        "Residual": y_test.values - y_pred
    })

    predictions.to_csv(
        os.path.join(OUTPUT_DIR, "predictions.csv"),
        index=False
    )

    # Actual vs predicted graph
    plt.figure(figsize=(7, 5))

    plt.scatter(y_test, y_pred, alpha=0.6)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Screen Time")
    plt.ylabel("Predicted Screen Time")
    plt.title("Actual vs Predicted Screen Time")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    )
    plt.close()

    # Residual analysis
    residuals = y_test.values - y_pred

    plt.figure(figsize=(7, 5))

    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted Screen Time")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "residual_plot.png")
    )
    plt.close()

    print("\nLinear regression completed!")


if __name__ == "__main__":
    train_model()
