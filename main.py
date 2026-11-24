
from data_cleaning import clean_data
from outlier_detection import detect_outliers
from eda import perform_eda
from linear_regression import train_model


def main():
    print("\n===================================")
    print(" PERSONAL SCREEN-TIME PROJECT")
    print("===================================")

    try:
        clean_data()
        detect_outliers()
        perform_eda()
        train_model()

        print("\n===================================")
        print(" ALL PROJECT TASKS COMPLETED!")
        print(" Check the output folder.")
        print("===================================")

    except Exception as error:
        print("\nPROJECT ERROR:", error)


if __name__ == "__main__":
    main()
