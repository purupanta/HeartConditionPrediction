# AppLib Class
# Author: Puru Panta (purupanta@uky.edu)
# Date: 11/30/2024

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# To load requirements.txt
import subprocess
import importlib.util

class AppLib:

    # Loading the requirements.txt
    @staticmethod
    # Function to load and install missing requirements
    def LoadRequirements(requirements_path):
        """Loads and installs missing packages from a requirements.txt file with Jupyter-style commands."""
        try:
            # Read the requirements file
            with open(requirements_path, "r") as file:
                required_packages = [
                    line.strip().replace("!pip", "").replace("install ", "").replace("--upgrade ", "").strip()
                    for line in file.readlines()
                    if line.strip()
                ]

            # Flatten multiple packages per line into a single list
            cleaned_packages = []
            for pkg in required_packages:
                cleaned_packages.extend(pkg.split())  # Splits lines like "scikit-learn imbalanced-learn"

            if not cleaned_packages:
                print("No valid packages found in the requirements file.")
                return

            # Check for missing packages
            missing_packages = [
                pkg for pkg in cleaned_packages
                if importlib.util.find_spec(pkg.split("==")[0]) is None
            ]

            # Install only the missing packages
            if missing_packages:
                print("Installing missing packages:", missing_packages)
                try:
                    subprocess.run(["pip", "install", "--upgrade"] + missing_packages, check=True)
                    print("Installation completed successfully!")
                    print("Please restart the runtime for changes to take effect.")
                except subprocess.CalledProcessError as e:
                    print(f"Error installing packages: {e}")
            else:
                print("All required packages are already installed!")

        except FileNotFoundError:
            print(f"Error: The file '{requirements_path}' was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    # Load the dataset
    @staticmethod
    def load_data(file_path, sheet_name, colsToLoad):
        # Load the dataset
        # Load all columns if colsToLoad is None or empty
        df_orig = '';
        if not colsToLoad:
            df_orig = pd.read_excel(file_path, sheet_name=sheet_name)  # Load full dataset
        else:
            df_orig = pd.read_excel(file_path, sheet_name=sheet_name, usecols=colsToLoad)  # Load only specified columns

        AppLib.print_datainfo(df_orig, 'Loaded, original data')
        return df_orig

    # Print dataset information
    @staticmethod
    def print_datainfo(data, flag):
        print(f'Data Size: {data.size}, Data Shape: {data.shape}, (Flag: {flag})')

    # Remove rows with null, blank, or whitespace
    @staticmethod
    def drop_rows_null(df_orig):
        df = df_orig.copy()
        df_drop_nulls = df[~df.apply(lambda row: any(col is None or str(col).strip() == '' for col in row), axis=1)]
        AppLib.print_datainfo(df_drop_nulls, 'Drop rows having null or blank items')
        return df_drop_nulls

    # Drop rows with negative numbers
    @staticmethod
    def drop_rows_neg_number(df_orig, predictors_and_targets):
        df = df_orig.copy()
        # Keep only rows where the target and predictors have no negative values
        data_cleaned = df[(df[predictors_and_targets] >= 0).all(axis=1)]

        # rows_to_drop = df.index[df.lt(0).any(axis=1)]
        # df_drop_neg = df.drop(rows_to_drop)

        AppLib.print_datainfo(data_cleaned, 'Drop rows having negative numbers')
        return data_cleaned

    # Binary encoding for a column
    @staticmethod
    def encode_bin(df_orig, column):
        df_encode = df_orig.copy()
        df_encode[column] = df[column].map({1: 1, 2: 0})
        return df_encode

    # Filter specific columns
    @staticmethod
    def col_filter(df_orig, columns):
        df = df_orig.copy()
        df_col_filtered = df.filter(items=columns)
        AppLib.print_datainfo(df_col_filtered, 'Drop un-necessary columns for this study')
        return df_col_filtered

    # Filter rows for cleaning
    @staticmethod
    def row_filter(df_orig, predictors_and_targets):
        df = df_orig.copy()
        df_null_filtered = AppLib.drop_rows_null(df)

        df_null_neg_filtered = AppLib.drop_rows_neg_number(df_null_filtered, predictors_and_targets)
        return df_null_neg_filtered

    # Data corrections and transformations
    @staticmethod
    def data_correction(df_orig):
        df = df_orig.copy()
        mappings = {
            'MedConditions_Diabetes': {1: 1, 2: 0},
            'MedConditions_HighBP': {1: 1, 2: 0},
            'MedConditions_LungDisease': {1: 1, 2: 0},
            'MedConditions_Depression': {1: 1, 2: 0},
            'EverHadCancer': {1: 1, 2: 0},
            'Deaf': {1: 1, 2: 0},
            'MedConditions_HeartCondition': {1: 1, 2: 0},
            'smokeStat': {3: 0, 1: 2, 2: 1},
            # 'smokeStat': {1: 'current', 2: 'former': , 3: 'never'},

            'SmokeNow': {1: 2, 2: 1, 3: 0},
            'Smoke100': {1: 1, 2: 0},


        }
        for column, mapping in mappings.items():
            if column in df.columns:  # Check if the column exists
                df[column] = df[column].map(mapping)
        return df

    # Define the target and predictors
    @staticmethod
    def define_target_predictors(df_cleaned, predictors, target):
        X = df_cleaned[predictors]
        y = df_cleaned[target]

        # If `targets` contains only one column, convert `y` to Series
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:  # Single-column DataFrame
                y = y.squeeze()  # Convert to Series
            else:
                raise ValueError("Target must be a single column or specified explicitly as a Series.")


        # Filter out rows where y has values other than 0 or 1
        valid_indices = y.isin([0, 1])
        X = X[valid_indices]
        y = y[valid_indices]

        # Check for unique values in the target
        print(f"Unique values in the target variable after filtering: {y.unique()}")

        return X, y

