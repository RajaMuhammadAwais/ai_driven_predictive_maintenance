
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

class AnomalyDetector:
    def __init__(self, data_path, features):
        self.data_path = data_path
        self.features = features
        self.scaler = StandardScaler()
        self.data = None
        self.scaled_data = None

    def load_and_preprocess_data(self):
        """Loads data and preprocesses it for anomaly detection."""
        try:
            df = pd.read_csv(self.data_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            self.data = df[self.features]
            self.scaled_data = self.scaler.fit_transform(self.data)
            print("Data loaded and scaled for anomaly detection.")

        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
        except Exception as e:
            print(f"An error occurred during data loading or preprocessing: {e}")

    def train_isolation_forest(self, contamination=0.01, random_state=42):
        """Trains an Isolation Forest model for anomaly detection."""
        if self.scaled_data is None:
            print("No data to train. Please load and preprocess data first.")
            return None

        model = IsolationForest(contamination=contamination, random_state=random_state)
        model.fit(self.scaled_data)
        print("Isolation Forest model trained successfully.")
        return model

    def predict_isolation_forest(self, model, data_to_predict=None):
        """Predicts anomalies using a trained Isolation Forest model.

        Args:
            model: Trained Isolation Forest model.
            data_to_predict (np.array): Data to predict anomalies on. If None, uses the loaded and scaled data.

        Returns:
            np.array: Anomaly scores (-1 for outlier, 1 for inlier).
        """
        if data_to_predict is None:
            data_to_predict = self.scaled_data

        if data_to_predict is None:
            print("No data to predict on.")
            return None

        predictions = model.predict(data_to_predict)
        return predictions

    def build_autoencoder(self, encoding_dim=16):
        """Builds an Autoencoder model for anomaly detection."""
        if self.scaled_data is None:
            print("No data to build autoencoder. Please load and preprocess data first.")
            return None

        input_dim = self.scaled_data.shape[1]

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="relu")(input_layer)
        decoder = Dense(input_dim, activation="sigmoid")(encoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        print("Autoencoder model built successfully.")
        autoencoder.summary()
        return autoencoder

    def train_autoencoder(self, model, epochs=50, batch_size=32, validation_split=0.2, patience=5):
        """Trains the Autoencoder model."""
        if self.scaled_data is None:
            print("No data to train. Please load and preprocess data first.")
            return None

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        print("Training Autoencoder...")
        history = model.fit(
            self.scaled_data, self.scaled_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        print("Autoencoder training complete.")
        return history

    def predict_autoencoder(self, model, data_to_predict=None, threshold=None):
        """Predicts anomalies using a trained Autoencoder model.

        Args:
            model: Trained Autoencoder model.
            data_to_predict (np.array): Data to predict anomalies on. If None, uses the loaded and scaled data.
            threshold (float): Threshold for anomaly detection. If None, uses the 95th percentile of reconstruction errors.

        Returns:
            tuple: (reconstruction_errors, anomaly_predictions)
        """
        if data_to_predict is None:
            data_to_predict = self.scaled_data

        if data_to_predict is None:
            print("No data to predict on.")
            return None, None

        reconstructed = model.predict(data_to_predict)
        reconstruction_errors = np.mean(np.square(data_to_predict - reconstructed), axis=1)

        if threshold is None:
            threshold = np.percentile(reconstruction_errors, 95)

        anomaly_predictions = (reconstruction_errors > threshold).astype(int)
        print(f"Autoencoder anomaly detection complete. Threshold: {threshold:.4f}")
        return reconstruction_errors, anomaly_predictions

if __name__ == "__main__":
    data_file = "/home/ubuntu/synthetic_predictive_maintenance_data.csv"
    features = ["cpu_load", "memory_usage", "disk_activity"]

    detector = AnomalyDetector(data_file, features)
    detector.load_and_preprocess_data()

    if detector.scaled_data is not None:
        # Train and test Isolation Forest
        print("\n--- Isolation Forest ---")
        iso_forest = detector.train_isolation_forest(contamination=0.05)
        iso_predictions = detector.predict_isolation_forest(iso_forest)
        print(f"Isolation Forest detected {np.sum(iso_predictions == -1)} anomalies out of {len(iso_predictions)} samples.")

        # Train and test Autoencoder
        print("\n--- Autoencoder ---")
        autoencoder = detector.build_autoencoder(encoding_dim=8)
        if autoencoder is not None:
            detector.train_autoencoder(autoencoder, epochs=20, batch_size=64)
            reconstruction_errors, ae_predictions = detector.predict_autoencoder(autoencoder)
            print(f"Autoencoder detected {np.sum(ae_predictions == 1)} anomalies out of {len(ae_predictions)} samples.")

            # Compare with actual failures
            df = pd.read_csv(data_file)
            actual_failures = df["failure_indicator"].values
            print(f"Actual failures in data: {np.sum(actual_failures == 1)}")

            # Calculate some basic metrics for comparison
            from sklearn.metrics import classification_report
            print("\nIsolation Forest Classification Report:")
            iso_binary = (iso_predictions == -1).astype(int)
            print(classification_report(actual_failures, iso_binary))

            print("\nAutoencoder Classification Report:")
            print(classification_report(actual_failures, ae_predictions))

