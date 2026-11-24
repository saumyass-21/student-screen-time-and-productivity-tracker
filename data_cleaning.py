
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "clean_smartphone_usage.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "1_data_cleaning")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_data():
    print("\n========== DATA CLEANING ==========")

    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Dataset not found. Place clean_smartphone_usage.csv "
            "in the project folder."
        )

    original_rows = len(df)

    report = []
    report.append(f"Original dataset shape: {df.shape}")

    # Check missing values
    report.append("\nMissing values before cleaning:")
    report.append(df.isnull().sum().to_string())

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    report.append(f"\nDuplicate rows removed: {duplicates}")

    # Convert numerical columns
    numeric_columns = [
        "age",
        "daily_screen_time_hours",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours",
        "notifications_per_day",
        "app_opens_per_day",
        "weekend_screen_time"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical text
    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in categorical_columns:
        df[col] = df[col].astype("string").str.strip()

    # Check invalid age values
    if "age" in df.columns:
        df.loc[
            (df["age"] < 0) | (df["age"] > 120),
            "age"
        ] = np.nan

    # Check invalid time values
    time_columns = [
        "daily_screen_time_hours",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours",
        "weekend_screen_time"
    ]

    for col in time_columns:
        if col in df.columns:
            df.loc[
                (df[col] < 0) | (df[col] > 24),
                col
            ] = np.nan

    # Check invalid count values
    count_columns = [
        "notifications_per_day",
        "app_opens_per_day"
    ]

    for col in count_columns:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Target column must not be missing
    target = "daily_screen_time_hours"

    if target not in df.columns:
        raise ValueError(f"Required column '{target}' is missing.")

    missing_targets = df[target].isna().sum()
    df = df.dropna(subset=[target])

    report.append(
        f"\nRows removed due to missing target: {missing_targets}"
    )

    # Impute numerical predictor columns with median
    for col in numeric_columns:
        if col in df.columns and col != target:
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical values with mode
    for col in categorical_columns:
        if col in df.columns and df[col].isna().any():
            mode_value = df[col].mode(dropna=True)

            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    # Final missing value check
    report.append("\nMissing values after cleaning:")
    report.append(df.isnull().sum().to_string())

    report.append(f"\nFinal dataset shape: {df.shape}")
    report.append(f"Original rows: {original_rows}")
    report.append(f"Final rows: {len(df)}")

    # Save cleaned dataset
    output_file = os.path.join(
        OUTPUT_DIR,
        "cleaned_dataset.csv"
    )

    df.to_csv(output_file, index=False)

    # Save cleaning report
    report_file = os.path.join(
        OUTPUT_DIR,
        "cleaning_report.txt"
    )

    with open(report_file, "w", encoding="utf-8") as file:
        file.write("\n".join(report))

    print("\nCleaning completed!")
    print("Cleaned dataset saved.")

    return df


if __name__ == "__main__":
    clean_data()
