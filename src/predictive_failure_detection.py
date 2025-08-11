
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Assuming TimeSeriesForecaster and AnomalyDetector classes are available
# For simplicity, I'll include their core logic here or import them if they were in separate files

# --- TimeSeriesForecaster Class (Simplified for integration) ---
class TimeSeriesForecaster:
    def __init__(self, features, target, timesteps=10):
        self.features = features
        self.target = target
        self.timesteps = timesteps
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self, df):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        data = df[self.features + [self.target]].values
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.timesteps, len(scaled_data)):
            X.append(scaled_data[i-self.timesteps:i, :])
            y.append(scaled_data[i, self.features.index(self.target)])
        return np.array(X), np.array(y)

    def build_model(self, model_type='lstm', units=50, dropout_rate=0.2):
        self.model = Sequential()
        if model_type == 'lstm':
            self.model.add(LSTM(units=units, return_sequences=True, input_shape=(self.timesteps, len(self.features) + 1)))
        elif model_type == 'gru':
            self.model.add(GRU(units=units, return_sequences=True, input_shape=(self.timesteps, len(self.features) + 1)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=units) if model_type == 'lstm' else GRU(units=units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=5):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

    def predict(self, input_data_scaled):
        predictions_scaled = self.model.predict(input_data_scaled)
        # Create a dummy array to inverse transform only the target feature
        dummy_array = np.zeros((input_data_scaled.shape[0], len(self.features) + 1))
        target_col_index = self.features.index(self.target)
        dummy_array[:, target_col_index] = predictions_scaled.flatten()
        predictions = self.scaler.inverse_transform(dummy_array)[:, target_col_index]
        return predictions

# --- AnomalyDetector Class (Simplified for integration) ---
class AnomalyDetector:
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        data = df[self.features].values
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def train_isolation_forest(self, scaled_data, contamination=0.01, random_state=42):
        model = IsolationForest(contamination=contamination, random_state=random_state)
        model.fit(scaled_data)
        return model

    def predict_isolation_forest(self, model, data_to_predict_scaled):
        return model.predict(data_to_predict_scaled)

    def build_autoencoder(self, input_dim, encoding_dim=16):
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="relu")(input_layer)
        decoder = Dense(input_dim, activation="sigmoid")(encoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder

    def train_autoencoder(self, model, scaled_data, epochs=50, batch_size=32, validation_split=0.2, patience=5):
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        model.fit(scaled_data, scaled_data, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping], verbose=0)

    def predict_autoencoder(self, model, data_to_predict_scaled, threshold=None):
        reconstructed = model.predict(data_to_predict_scaled)
        reconstruction_errors = np.mean(np.square(data_to_predict_scaled - reconstructed), axis=1)
        if threshold is None:
            threshold = np.percentile(reconstruction_errors, 95)
        anomaly_predictions = (reconstruction_errors > threshold).astype(int)
        return reconstruction_errors, anomaly_predictions

# --- Main Integration Logic ---
if __name__ == "__main__":
    data_file = "/home/ubuntu/synthetic_predictive_maintenance_data.csv"
    features = ["cpu_load", "memory_usage", "disk_activity"]
    target_for_forecasting = "cpu_load" # Example: predict CPU load
    timesteps = 60

    # Load the full dataset
    df = pd.read_csv(data_file)

    # --- Time Series Forecasting ---
    print("\n--- Time Series Forecasting (LSTM) ---")
    forecaster = TimeSeriesForecaster(features, target_for_forecasting, timesteps)
    X_ts, y_ts = forecaster.prepare_data(df.copy()) # Use a copy to avoid modifying original df

    # Split for forecasting model training/validation
    X_train_ts, X_val_ts, y_train_ts, y_val_ts = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42, shuffle=False)

    forecaster.build_model(model_type='lstm', units=64)
    forecaster.train_model(X_train_ts, y_train_ts, X_val_ts, y_val_ts, epochs=10, batch_size=64)

    # Get predictions for the test set
    predicted_metrics_scaled = forecaster.model.predict(X_val_ts)

    # For anomaly detection, we need the full feature set, not just the target
    # We'll use the actual values from the test set for simplicity in this example
    # In a real scenario, you might use predicted future values for anomaly detection
    # or combine current real-time data with historical patterns.
    anomaly_detection_data_df = df.iloc[-len(X_val_ts):].copy() # Use corresponding part of original df

    # --- Anomaly Detection ---
    print("\n--- Anomaly Detection (Isolation Forest & Autoencoder) ---")
    anomaly_detector = AnomalyDetector(features)
    scaled_anomaly_data = anomaly_detector.prepare_data(anomaly_detection_data_df)

    # Isolation Forest
    iso_forest_model = anomaly_detector.train_isolation_forest(scaled_anomaly_data, contamination=0.05)
    iso_predictions = anomaly_detector.predict_isolation_forest(iso_forest_model, scaled_anomaly_data)
    iso_anomalies = (iso_predictions == -1).astype(int)

    # Autoencoder
    input_dim_ae = scaled_anomaly_data.shape[1]
    autoencoder_model = anomaly_detector.build_autoencoder(input_dim_ae, encoding_dim=8)
    anomaly_detector.train_autoencoder(autoencoder_model, scaled_anomaly_data, epochs=20, batch_size=64)
    reconstruction_errors, ae_anomalies = anomaly_detector.predict_autoencoder(autoencoder_model, scaled_anomaly_data)

    # --- Combine Predictions for Failure Detection ---
    print("\n--- Combined Failure Prediction ---")
    # A simple strategy: if either model detects an anomaly, flag as potential failure
    # In a real system, this would involve more sophisticated fusion and thresholding
    combined_predictions = np.zeros(len(iso_anomalies))
    for i in range(len(iso_anomalies)):
        if iso_anomalies[i] == 1 or ae_anomalies[i] == 1:
            combined_predictions[i] = 1

    # Evaluate combined predictions against actual failures in the test set portion
    actual_failures_test_set = df["failure_indicator"].iloc[-len(X_val_ts):].values

    print("\nClassification Report for Combined Model:")
    print(classification_report(actual_failures_test_set, combined_predictions))

    print("\nSample of Combined Predictions (first 10):\n", combined_predictions[:10])
    print("Sample of Actual Failures (first 10):\n", actual_failures_test_set[:10])

    print(f"Total actual failures in test set: {np.sum(actual_failures_test_set)}")
    print(f"Total combined model detected anomalies: {np.sum(combined_predictions)}")



