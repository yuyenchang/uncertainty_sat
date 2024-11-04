# rnn_model.py

import numpy as np
from tensorflow.keras import layers, models

class RNNModel:
    def __init__(self, input_shape):
        """
        Initialize the Recurrent Neural Network (RNN) model using LSTM layers.
        
        Model Design Decisions:
        - LSTM layers are used for their ability to capture sequential dependencies, crucial for time-series orbit prediction.
        - Dropout layers are included to prevent overfitting and to enable uncertainty quantification by simulating different neural network configurations.
        - A TimeDistributed Dense layer with 3 units at the output layer is used to simultaneously predict X, Y, and Z coordinates.
        """
        self.model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),  # First LSTM layer with 64 units
            layers.Dropout(0.2),  # Dropout layer to help with generalization and uncertainty quantification
            layers.LSTM(32, return_sequences=True),  # Second LSTM layer with 32 units for deeper sequence learning
            layers.Dropout(0.2),  # Additional dropout layer to improve robustness
            layers.TimeDistributed(layers.Dense(3))  # Output layer to predict 3D coordinates over time steps
        ])
        # Compile the model with the Adam optimizer and Mean Squared Error (MSE) loss.
        # - MSE is appropriate for this task as it penalizes larger errors more significantly, aiding precise 3D position prediction.
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.33):
        """
        Train the RNN model on provided data.
        
        Training Parameters:
        - Batch size of 32 balances memory efficiency and training speed.
        - Validation split of 33% helps monitor model generalization on unseen data.
        - The default of 100 epochs allows the model to learn data patterns effectively, though this may be adjusted based on convergence behavior.
        """
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)

    def predict_with_uncertainty(self, input_data, n_iter=100, X_min=None, X_max=None):
        """
        Generate predictions with uncertainty by enabling dropout during inference.
        
        Approach:
        - Monte Carlo Dropout: By keeping dropout active during inference, we generate a distribution of predictions.
        - This approach quantifies uncertainty by calculating the mean and standard deviation across multiple predictions.
        
        Arguments:
        - n_iter: Number of iterations to run with dropout enabled for uncertainty estimation.
        - X_min, X_max: Optional normalization parameters for denormalizing outputs to original scale.
        
        Returns:
        - prediction_mean: The mean of the generated predictions, representing the expected outcome.
        - prediction_uncertainty: The standard deviation, representing uncertainty in the predictions.
        """
        predictions = [self.model(input_data, training=True) for _ in range(n_iter)]  # Generate predictions with dropout
        predictions = np.stack(predictions, axis=0)  # Stack predictions to calculate statistics
        prediction_mean = predictions.mean(axis=0)  # Expected outcome
        prediction_uncertainty = predictions.std(axis=0)  # Prediction uncertainty

        # Denormalize if min/max values are provided, converting predictions back to original scale.
        if X_min is not None and X_max is not None:
            prediction_mean = prediction_mean * (X_max - X_min) + X_min  # Denormalize mean
            prediction_uncertainty = prediction_uncertainty * (X_max - X_min)  # Denormalize uncertainty

        return prediction_mean, prediction_uncertainty  # Return the predictions and associated uncertainties

