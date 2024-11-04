# satellite_predictor.py

import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

class SatellitePredictor:
    def __init__(self, tle_line1, tle_line2, start_time, time_interval, num_intervals, uncertainty_factor=0.1, samples=100):
        """
        Initialize SatellitePredictor with TLE data, prediction time settings, and uncertainty parameters.

        Parameters:
        - tle_line1 (str): First line of the Two-Line Element (TLE) data.
        - tle_line2 (str): Second line of the Two-Line Element (TLE) data.
        - start_time (datetime): The starting time for generating predictions.
        - time_interval (timedelta): The interval between each position prediction.
        - num_intervals (int): Total number of prediction intervals.
        - uncertainty_factor (float): Factor controlling the magnitude of added noise to simulate real-world uncertainties.
        - samples (int): Number of samples to represent uncertainty in position predictions.
        """
        self.satellite = Satrec.twoline2rv(tle_line1, tle_line2)
        self.start_time = start_time
        self.time_interval = time_interval
        self.num_intervals = num_intervals
        self.uncertainty_factor = uncertainty_factor
        self.samples = samples

    def generate_orbital_data(self):
        """
        Generate satellite position data over defined intervals using the SGP4 model.
     
        Returns:
        - positions (numpy.ndarray): Array of satellite positions at each interval.
        - times (list): List of timestamps corresponding to each predicted position.
        """
        positions, times = [], []
        for i in range(self.num_intervals):
            current_time = self.start_time + i * self.time_interval  # Calculate time for current interval
            jd, fr = jday(current_time.year, current_time.month, current_time.day,
                          current_time.hour, current_time.minute, current_time.second)  # Convert to Julian date
            error_code, position, _ = self.satellite.sgp4(jd, fr)  # Get position using SGP4
            if error_code == 0:  # If calculation is successful
                positions.append(position)
                times.append(current_time)
        return np.array(positions), times

    def add_uncertainty(self, positions):
        """
        Add uncertainty to position predictions by introducing Gaussian noise.
        
        Parameters:
        - positions (numpy.ndarray): Array of satellite positions.

        Returns:
        - numpy.ndarray: Array of position samples with added noise, simulating prediction uncertainties.
        """
        return positions + np.random.normal(0, self.uncertainty_factor * np.abs(positions),
                                            (self.samples, self.num_intervals, 3))

    def preprocess_data(self, positions_with_uncertainty):
        """
        Prepare data for model training by creating features and targets, then normalizing them.
     
        Returns:
        - X (numpy.ndarray): Normalized features.
        - y (numpy.ndarray): Normalized targets.
        - (X_min, X_max) (tuple): Min-Max normalization factors for features.
        - (y_min, y_max) (tuple): Min-Max normalization factors for targets.
         """
        X = positions_with_uncertainty[:, :-1]
        y = positions_with_uncertainty[:, 1:]

        X_min, X_max = np.min(X, axis=(0, 1)), np.max(X, axis=(0, 1))
        y_min, y_max = np.min(y, axis=(0, 1)), np.max(y, axis=(0, 1))

        X = (X - X_min) / (X_max - X_min)
        y = (y - y_min) / (y_max - y_min)

        return X, y, (X_min, X_max), (y_min, y_max)

