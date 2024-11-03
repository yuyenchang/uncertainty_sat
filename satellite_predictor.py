# satellite_predictor.py

import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

class SatellitePredictor:
    def __init__(self, tle_line1, tle_line2, start_time, time_interval, num_intervals, uncertainty_factor=0.1, samples=100):
        # Initialize the satellite object using Two-Line Element (TLE) data
        self.satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        # Set the starting time for the predictions
        self.start_time = start_time
        # Define the time interval between predictions
        self.time_interval = time_interval
        # Set the number of intervals for which predictions are to be made
        self.num_intervals = num_intervals
        # Define the uncertainty factor for adding noise to the predictions
        self.uncertainty_factor = uncertainty_factor
        # Set the number of samples to generate uncertainty data
        self.samples = samples

    def generate_orbital_data(self):
        # Generate orbital position data over specified time intervals
        positions, times = [], []
        for i in range(self.num_intervals):
            # Calculate the current time for this interval
            current_time = self.start_time + i * self.time_interval
            # Convert the current time to Julian Date (JD) and fraction
            jd, fr = jday(current_time.year, current_time.month, current_time.day,
                          current_time.hour, current_time.minute, current_time.second)
            # Get the satellite's position at the specified JD and fraction
            error_code, position, _ = self.satellite.sgp4(jd, fr)
            if error_code == 0:  # Check if the calculation was successful (error code 0 means success)
                positions.append(position)  # Store the position
                times.append(current_time)    # Store the corresponding time
        return np.array(positions), times  # Return the positions as a numpy array and the times

    def add_uncertainty(self, positions):
        # Add uncertainty to the positions by generating random noise
        return positions + np.random.normal(0, self.uncertainty_factor * np.abs(positions), (self.samples, self.num_intervals, 3))

    def preprocess_data(self, positions_with_uncertainty):
        # Split the data into features (X) and targets (y) for training
        X = positions_with_uncertainty[:, :-1]  # All but the last position as features
        y = positions_with_uncertainty[:, 1:]    # All but the first position as targets

        # Normalize the data using Min-Max normalization
        X_min, X_max = np.min(X, axis=(0, 1)), np.max(X, axis=(0, 1))  # Get min and max for features
        y_min, y_max = np.min(y, axis=(0, 1)), np.max(y, axis=(0, 1))  # Get min and max for targets

        X = (X - X_min) / (X_max - X_min)  # Normalize features
        y = (y - y_min) / (y_max - y_min)  # Normalize targets

        return X, y, (X_min, X_max), (y_min, y_max)  # Return normalized data and normalization factors
