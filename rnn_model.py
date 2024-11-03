# rnn_model.py

import numpy as np
from tensorflow.keras import layers, models

class RNNModel:
    def __init__(self, input_shape):
        # Initialize the Recurrent Neural Network (RNN) model using LSTM layers
        self.model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),  # First LSTM layer with 64 units
            layers.Dropout(0.2),  # Dropout layer to prevent overfitting
            layers.LSTM(32, return_sequences=True),  # Second LSTM layer with 32 units
            layers.Dropout(0.2),  # Another dropout layer
            layers.TimeDistributed(layers.Dense(3))  # Output layer to predict 3D coordinates
        ])
        # Compile the model with Adam optimizer and Mean Squared Error loss function
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.33):
        # Train the RNN model on the provided data
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)

    def predict_with_uncertainty(self, input_data, n_iter=100, X_min=None, X_max=None):
        # Predict the output with uncertainty by enabling dropout during inference
        predictions = [self.model(input_data, training=True) for _ in range(n_iter)]  # Generate predictions
        predictions = np.stack(predictions, axis=0)  # Stack predictions to calculate mean and std
        prediction_mean = predictions.mean(axis=0)  # Mean of the predictions
        prediction_uncertainty = predictions.std(axis=0)  # Standard deviation for uncertainty

        # Denormalize predictions if min/max values are provided
        if X_min is not None and X_max is not None:
            prediction_mean = prediction_mean * (X_max - X_min) + X_min  # Denormalize mean predictions
            prediction_uncertainty = prediction_uncertainty * (X_max - X_min)  # Denormalize uncertainty

        return prediction_mean, prediction_uncertainty  # Return mean predictions and uncertainty

