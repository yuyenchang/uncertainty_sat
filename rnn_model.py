# rnn_model.py

import numpy as np
from tensorflow.keras import layers, models

class RNNModel:
    def __init__(self, input_shape):
        """
        Initialize the Recurrent Neural Network (RNN) model using LSTM layers.
        
        LSTM layers capture sequential dependencies, which are crucial for time-series orbit prediction. 
        Dropout layers help prevent overfitting and enable uncertainty quantification by simulating different neural network configurations. 
        A TimeDistributed Dense layer with 3 units at the output layer predicts X, Y, and Z coordinates simultaneously.
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
        
        Parameters:
        - X (numpy.ndarray): Input features for training.
        - y (numpy.ndarray): Target labels corresponding to the input features.
        - epochs (int): Number of training epochs (default is 100).
        - batch_size (int): Number of samples per gradient update (default is 32), balancing memory efficiency and training speed.
        - validation_split (float): Fraction of the training data to be used as validation data (default is 0.33), helping monitor generalization on unseen data.
        """
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)

    def predict_with_uncertainty(self, input_data, n_iter=100, X_min=None, X_max=None):
        """
        Generate predictions with uncertainty by enabling dropout during inference.
        
        Parameters:
        - input_data (numpy.ndarray): Input data for which predictions are to be made.
        - n_iter (int): Number of iterations to run with dropout enabled for uncertainty estimation (default is 100).
        - X_min (float, optional): Minimum value for denormalizing outputs to the original scale.
        - X_max (float, optional): Maximum value for denormalizing outputs to the original scale.
        
        Returns:
        - prediction_mean (numpy.ndarray): The mean of the generated predictions, representing the expected outcome.
        - prediction_uncertainty (numpy.ndarray): The standard deviation of the predictions, indicating the uncertainty.
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

