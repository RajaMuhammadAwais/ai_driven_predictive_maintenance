
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

    def preprocess_data(self, handle_missing='mean', scale_features=True):
        """Performs basic preprocessing on the ingested data.

        Args:
            handle_missing (str): Strategy for handling missing values. 'mean' for mean imputation, 'median' for median imputation, 'drop' to drop rows with missing values.
            scale_features (bool): Whether to scale numerical features using StandardScaler.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        if self.data is None:
            print("No data to preprocess. Please ingest data first.")
            return None

        processed_data = self.data.copy()

        # Handle missing values
        if handle_missing == 'mean':
            for col in processed_data.select_dtypes(include=['number']).columns:
                processed_data[col].fillna(processed_data[col].mean(), inplace=True)
        elif handle_missing == 'median':
            for col in processed_data.select_dtypes(include=['number']).columns:
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
        elif handle_missing == 'drop':
            processed_data.dropna(inplace=True)
        else:
            print("Invalid handle_missing strategy. Supported: 'mean', 'median', 'drop'.")

        # Scale numerical features
        if scale_features:
            numerical_cols = processed_data.select_dtypes(include=['number']).columns
            if not numerical_cols.empty:
                scaler = StandardScaler()
                processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])
                print("Numerical features scaled.")
            else:
                print("No numerical features to scale.")

        print("Data preprocessing complete.")
        return processed_data

if __name__ == '__main__':
    # Example Usage (assuming a 'sample_data.csv' exists in the same directory)
    # For demonstration, let's create a dummy CSV file
    dummy_data = {
        'feature1': [10, 20, None, 40, 50],
        'feature2': [1.1, 2.2, 3.3, 4.4, None],
        'category': ['A', 'B', 'A', 'C', 'B']
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = '/home/ubuntu/src/sample_data.csv'
    dummy_df.to_csv(dummy_csv_path, index=False)

    print(f"Dummy data saved to {dummy_csv_path}")

    ingestor = DataIngestionAndPreprocessing(dummy_csv_path)
    raw_data = ingestor.ingest_data()

    if raw_data is not None:
        print("\nRaw Data Head:")
        print(raw_data.head())

        preprocessed_data = ingestor.preprocess_data(handle_missing='mean', scale_features=True)
        if preprocessed_data is not None:
            print("\nPreprocessed Data Head:")
            print(preprocessed_data.head())

        preprocessed_data_no_scale = ingestor.preprocess_data(handle_missing='drop', scale_features=False)
        if preprocessed_data_no_scale is not None:
            print("\nPreprocessed Data (no scaling, dropped missing) Head:")
            print(preprocessed_data_no_scale.head())



