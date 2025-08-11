
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class TimeSeriesForecaster:
    def __init__(self, data_path, features, target, timesteps=10):
        self.data_path = data_path
        self.features = features
        self.target = target
        self.timesteps = timesteps
        self.scaler = MinMaxScaler()
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_and_preprocess_data(self):
        """Loads data and preprocesses it for time series forecasting."""
        try:
            df = pd.read_csv(self.data_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            # Select features and target
            data = df[self.features + [self.target]].values

            # Scale the data
            scaled_data = self.scaler.fit_transform(data)

            # Create sequences for time series forecasting
            X, y = [], []
            for i in range(self.timesteps, len(scaled_data)):
                X.append(scaled_data[i-self.timesteps:i, :])
                y.append(scaled_data[i, self.features.index(self.target)]) # Target is the scaled value of the target feature

            X = np.array(X)
            y = np.array(y)

            # Split data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            print("Data loaded and preprocessed successfully.")
            print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
        except Exception as e:
            print(f"An error occurred during data loading or preprocessing: {e}")

    def build_model(self, model_type='lstm', units=50, dropout_rate=0.2):
        """Builds the LSTM or GRU model."""
        self.model = Sequential()
        if model_type == 'lstm':
            self.model.add(LSTM(units=units, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        elif model_type == 'gru':
            self.model.add(GRU(units=units, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        else:
            raise ValueError("model_type must be 'lstm' or 'gru'")

        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=units)) # Second LSTM/GRU layer, only return last output
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1)) # Output layer for regression

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print(f"{model_type.upper()} model built successfully.")
        self.model.summary()

    def train_model(self, epochs=50, batch_size=32, patience=5):
        """Trains the built model."""
        if self.model is None:
            print("Model not built. Please call build_model() first.")
            return

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        print("Training model...")
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        print("Model training complete.")
        return history

    def evaluate_model(self):
        """Evaluates the trained model on the test set."""
        if self.model is None:
            print("Model not built or trained. Please call build_model() and train_model() first.")
            return

        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Model evaluation - Test Loss: {loss:.4f}")
        return loss

    def predict(self, input_data):
        """Makes predictions using the trained model.

        Args:
            input_data (np.array): New data for prediction. Must be scaled and reshaped (num_samples, timesteps, num_features).

        Returns:
            np.array: Inverse transformed predictions.
        """
        if self.model is None:
            print("Model not built or trained. Cannot make predictions.")
            return None

        # Predictions are on scaled data, inverse transform them
        # Create a dummy array to inverse transform only the target feature
        dummy_array = np.zeros((input_data.shape[0], len(self.features) + 1)) # +1 for target
        target_col_index = self.features.index(self.target)

        predictions_scaled = self.model.predict(input_data)

        # Place scaled predictions into the correct column of the dummy array
        dummy_array[:, target_col_index] = predictions_scaled.flatten()

        # Inverse transform the dummy array to get actual predictions
        predictions = self.scaler.inverse_transform(dummy_array)[:, target_col_index]

        return predictions

if __name__ == "__main__":
    data_file = "synthetic_predictive_maintenance_data.csv"
    features = ["cpu_load", "memory_usage", "disk_activity"]
    target = "cpu_load" # Predicting CPU load as an example
    timesteps = 60 # Using 60 minutes of data to predict next

    forecaster = TimeSeriesForecaster(data_file, features, target, timesteps)
    forecaster.load_and_preprocess_data()

    if forecaster.X_train is not None:
        # Build and train LSTM model
        forecaster.build_model(model_type='lstm', units=64)
        forecaster.train_model(epochs=10, batch_size=64)
        forecaster.evaluate_model()

        # Example prediction (using a slice of test data)
        sample_input = forecaster.X_test[0:1]
        predicted_value = forecaster.predict(sample_input)
        print(f"\nSample Input Shape for Prediction: {sample_input.shape}")
        print(f"Predicted {target} for next timestep: {predicted_value[0]:.4f}")

        # Build and train GRU model
        forecaster.build_model(model_type='gru', units=64)
        forecaster.train_model(epochs=10, batch_size=64)
        forecaster.evaluate_model()

        # Example prediction for GRU
        predicted_value_gru = forecaster.predict(sample_input)
        print(f"\nSample Input Shape for Prediction (GRU): {sample_input.shape}")
        print(f"Predicted {target} for next timestep (GRU): {predicted_value_gru[0]:.4f}")



