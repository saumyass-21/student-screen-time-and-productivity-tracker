
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    BASE_DIR, "output", "1_data_cleaning", "cleaned_dataset.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "3_eda")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def perform_eda():
    print("\n========== EXPLORATORY DATA ANALYSIS ==========")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "Cleaned dataset not found. Run data cleaning first."
        )

    df = pd.read_csv(DATA_FILE)

    # 1. Histograms
    histogram_columns = [
        "age",
        "daily_screen_time_hours",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours"
    ]

    for col in histogram_columns:
        if col not in df.columns:
            continue

        plt.figure(figsize=(7, 4))

        sns.histplot(
            df[col].dropna(),
            bins=20,
            kde=True
        )

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"histogram_{col}.png")
        )
        plt.close()

    # 2. Correlation heatmap
    numerical_df = df.select_dtypes(include="number")
    correlation = numerical_df.corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    )
    plt.close()

    # 3. Categorical distribution
    categorical_columns = [
        "gender",
        "stress_level",
        "academic_work_impact",
        "addiction_level",
        "addicted_label"
    ]

    for col in categorical_columns:
        if col not in df.columns:
            continue

        counts = df[col].value_counts()

        plt.figure(figsize=(7, 4))
        counts.plot(kind="bar")

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Number of Students")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"distribution_{col}.png")
        )
        plt.close()

    # 4. Average screen time by category
    comparison_columns = [
        "gender",
        "stress_level",
        "academic_work_impact",
        "addiction_level"
    ]

    for col in comparison_columns:
        if col not in df.columns:
            continue

        averages = df.groupby(
            col,
            dropna=False
        )["daily_screen_time_hours"].mean()

        plt.figure(figsize=(7, 4))
        averages.plot(kind="bar")

        plt.title(f"Average Screen Time by {col}")
        plt.xlabel(col)
        plt.ylabel("Average Screen Time (Hours)")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"average_screen_time_{col}.png")
        )
        plt.close()

    print("\nEDA completed!")
    print("All EDA graphs have been saved.")


if __name__ == "__main__":
    perform_eda()
