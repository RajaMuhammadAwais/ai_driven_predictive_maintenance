# Data Ingestion and Preprocessing Module Documentation

## Introduction

This document provides comprehensive documentation for the `DataIngestionAndPreprocessing` Python module, designed as a foundational component for the AI-Driven Predictive Maintenance and Dynamic Resource Optimization system. This module handles the crucial tasks of ingesting raw data from CSV files and performing essential preprocessing steps, including missing value imputation and feature scaling. Clean and well-prepared data is paramount for the accuracy and effectiveness of any subsequent machine learning models in the predictive maintenance pipeline.

## Module Overview

The `DataIngestionAndPreprocessing` class encapsulates the functionalities required for data loading and initial transformation. It is built using popular Python libraries such as `pandas` for data manipulation and `scikit-learn` for preprocessing utilities. The module is designed to be flexible, allowing different strategies for handling missing values and an option to scale numerical features.

## Class: `DataIngestionAndPreprocessing`

### `__init__(self, file_path)`

**Purpose**: Initializes the `DataIngestionAndPreprocessing` class with the path to the data file.

**Parameters**:
- `file_path` (str): The absolute or relative path to the CSV file from which data will be ingested.

**Attributes Initialized**:
- `self.file_path` (str): Stores the provided file path.
- `self.data` (pandas.DataFrame): Will store the ingested data. Initialized to `None`.

### `ingest_data(self)`

**Purpose**: Ingests data from the specified CSV file into a pandas DataFrame.

**Returns**:
- `pandas.DataFrame`: The ingested data if successful.
- `None`: If the file is not found or an error occurs during ingestion.

**Error Handling**:
- Catches `FileNotFoundError` if the `file_path` does not point to an existing file.
- Catches general `Exception` for other potential issues during data reading (e.g., malformed CSV).

### `preprocess_data(self, handle_missing='mean', scale_features=True)`

**Purpose**: Performs basic preprocessing on the ingested data. This method handles missing values and optionally scales numerical features.

**Parameters**:
- `handle_missing` (str, optional): Strategy for handling missing numerical values. Supported options are:
    - `'mean'`: Imputes missing numerical values with the mean of their respective columns.
    - `'median'`: Imputes missing numerical values with the median of their respective columns.
    - `'drop'`: Drops rows that contain any missing values.
    - Default is `'mean'`.
- `scale_features` (bool, optional): A boolean flag indicating whether to scale numerical features using `StandardScaler` from `scikit-learn`. Default is `True`.

**Returns**:
- `pandas.DataFrame`: The preprocessed data.
- `None`: If no data has been ingested prior to calling this method.

**Preprocessing Steps**:
1.  **Missing Value Handling**: Based on the `handle_missing` parameter, numerical columns are imputed or rows with missing values are dropped. Non-numerical columns are not affected by imputation.
2.  **Feature Scaling**: If `scale_features` is `True`, numerical columns are scaled using `StandardScaler`. This transforms the data to have a mean of 0 and a standard deviation of 1, which is beneficial for many machine learning algorithms.

## Usage Example

Below is an example demonstrating how to use the `DataIngestionAndPreprocessing` module. This example assumes a dummy CSV file named `sample_data.csv` is available in the same directory as the script.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataIngestionAndPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def ingest_data(self):
        """Ingests data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data ingested successfully from {self.file_path}")
            return self.data
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred during data ingestion: {e}")
            return None

    def preprocess_data(self, handle_missing=\'mean\', scale_features=True):
        """Performs basic preprocessing on the ingested data.

        Args:
            handle_missing (str): Strategy for handling missing values. \'mean\' for mean imputation, \'median\' for median imputation, \'drop\' to drop rows with missing values.
            scale_features (bool): Whether to scale numerical features using StandardScaler.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        if self.data is None:
            print("No data to preprocess. Please ingest data first.")
            return None

        processed_data = self.data.copy()

        # Handle missing values
        if handle_missing == \'mean\':
            for col in processed_data.select_dtypes(include=[\'number\']).columns:
                processed_data[col].fillna(processed_data[col].mean(), inplace=True)
        elif handle_missing == \'median\':
            for col in processed_data.select_dtypes(include=[\'number\']).columns:
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
        elif handle_missing == \'drop\':
            processed_data.dropna(inplace=True)
        else:
            print("Invalid handle_missing strategy. Supported: \'mean\', \'median\', \'drop\'.")

        # Scale numerical features
        if scale_features:
            numerical_cols = processed_data.select_dtypes(include=[\'number\']).columns
            if not numerical_cols.empty:
                scaler = StandardScaler()
                processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])
                print("Numerical features scaled.")
            else:
                print("No numerical features to scale.")

        print("Data preprocessing complete.")
        return processed_data

if __name__ == \'__main__\':
    # Example Usage (assuming a \'sample_data.csv\' exists in the same directory)
    # For demonstration, let\'s create a dummy CSV file
    dummy_data = {
        \'feature1\': [10, 20, None, 40, 50],
        \'feature2\': [1.1, 2.2, 3.3, 4.4, None],
        \'category\': [\'A\', \'B\', \'A\', \'C\', \'B\']
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = \'/home/ubuntu/src/sample_data.csv\'
    dummy_df.to_csv(dummy_csv_path, index=False)

    print(f"Dummy data saved to {dummy_csv_path}")

    ingestor = DataIngestionAndPreprocessing(dummy_csv_path)
    raw_data = ingestor.ingest_data()

    if raw_data is not None:
        print("\nRaw Data Head:")
        print(raw_data.head())

        preprocessed_data = ingestor.preprocess_data(handle_missing=\'mean\', scale_features=True)
        if preprocessed_data is not None:
            print("\nPreprocessed Data Head:")
            print(preprocessed_data.head())

        preprocessed_data_no_scale = ingestor.preprocess_data(handle_missing=\'drop\', scale_features=False)
        if preprocessed_data_no_scale is not None:
            print("\nPreprocessed Data (no scaling, dropped missing) Head:")
            print(preprocessed_data_no_scale.head())

```

## Conclusion

The `DataIngestionAndPreprocessing` module provides a robust and flexible solution for handling the initial stages of data preparation for the AI-Driven Predictive Maintenance and Dynamic Resource Optimization system. By ensuring data quality and consistency, it lays a strong foundation for the subsequent machine learning and analytical tasks.

