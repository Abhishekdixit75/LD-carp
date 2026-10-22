"""
Practical 16: Time-Series Forecasting Using LSTM
Forecasts future temperature from past 24 hourly readings using LSTM; evaluates MAE & RMSE.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
tf.random.set_seed(42)


def experiment_16_time_series_lstm():
    print("=" * 65)
    print("PRACTICAL 16: TIME-SERIES FORECASTING USING LSTM")
    print("Forecasting Temperature from Previous 24 Hourly Readings")
    print("=" * 65)

    csv_candidates = ["jena_climate_2009_2016.csv", "jena_climate.csv", "temperature.csv", "climate.csv"]
    csv_file = next((f for f in csv_candidates if os.path.exists(f)), None)

    if csv_file:
        print(f"[Data Source] Found local CSV file: '{csv_file}' -> Training on CSV time series...")
        df = pd.read_csv(csv_file)
        temp_cols = [c for c in df.columns if "T (degC)" in c or "temp" in c.lower() or "temperature" in c.lower()]
        target_col = temp_cols[0] if temp_cols else df.select_dtypes(include=[np.number]).columns[0]
        temperature = df[target_col].dropna().values[:2500].astype("float32")
    else:
        print("[Data Source] No local CSV found -> Generating synthetic hourly temperature series (Jena climate)...")
        total_hours = 2000
        t = np.arange(total_hours)
        temperature = 15.0 + 8.0 * np.sin(2 * np.pi * t / 24.0) + 0.005 * t + np.random.normal(0, 1.0, total_hours)
        temperature = temperature.astype("float32")

    sequence_length = 24
    # Normalize
    split_point = int(len(temperature) * 0.8)
    mean, std = temperature[:split_point].mean(), temperature[:split_point].std()
    norm_temp = (temperature - mean) / (std + 1e-7)

    # Create sliding windows
    x, y = [], []
    for i in range(len(norm_temp) - sequence_length):
        x.append(norm_temp[i : i + sequence_length])
        y.append(norm_temp[i + sequence_length])
    x = np.array(x)[..., np.newaxis]
    y = np.array(y)

    train_split = int(len(x) * 0.8)
    x_train, x_test = x[:train_split], x[train_split:]
    y_train, y_test = y[:train_split], y[train_split:]

    # Build LSTM Model
    model = keras.Sequential([
        keras.Input(shape=(sequence_length, 1)),
        layers.LSTM(32),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    print("\nTraining LSTM Forecasting Model...")
    model.fit(x_train, y_train, epochs=4, batch_size=64, validation_split=0.1, verbose=1)

    # Predict & Denormalize
    predictions_norm = model.predict(x_test, verbose=0).ravel()
    predictions = predictions_norm * std + mean
    actuals = y_test * std + mean

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    print("\n" + "=" * 65)
    print("FORECASTING EVALUATION METRICS:")
    print("-" * 65)
    print(f"Mean Absolute Error (MAE):     {mae:.3f} deg C")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} deg C")
    print(f"\nSample Predictions (Next 5 Hours):")
    for i in range(5):
        print(f"Hour +{i+1}: Actual = {actuals[i]:.2f} deg C, Predicted = {predictions[i]:.2f} deg C")
    print("=" * 65)


if __name__ == "__main__":
    experiment_16_time_series_lstm()
