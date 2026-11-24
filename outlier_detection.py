
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    BASE_DIR, "output", "1_data_cleaning", "cleaned_dataset.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "2_outlier_detection")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_outliers():
    print("\n========== OUTLIER DETECTION ==========")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "Cleaned dataset not found. Run data cleaning first."
        )

    df = pd.read_csv(DATA_FILE)

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    results = []

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outliers = df[
            (df[col] < lower_limit) |
            (df[col] > upper_limit)
        ]

        results.append({
            "Column": col,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Limit": lower_limit,
            "Upper Limit": upper_limit,
            "Outlier Count": len(outliers)
        })

        print(f"\nColumn: {col}")
        print("Lower Limit:", lower_limit)
        print("Upper Limit:", upper_limit)
        print("Outlier Count:", len(outliers))

        # Create boxplot
        plt.figure(figsize=(7, 4))
        plt.boxplot(df[col].dropna(), vert=False)
        plt.title(f"Outlier Detection - {col}")
        plt.xlabel(col)
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_DIR, f"boxplot_{col}.png")
        )
        plt.close()

    # Save outlier report
    report = pd.DataFrame(results)

    report.to_csv(
        os.path.join(OUTPUT_DIR, "outlier_report.csv"),
        index=False
    )

    print("\nOutlier detection completed!")


if __name__ == "__main__":
    detect_outliers()
