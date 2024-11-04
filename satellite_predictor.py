# satellite_predictor.py

import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta

class SatellitePredictor:
    def __init__(self, tle_line1, tle_line2, start_time, time_interval, num_intervals, uncertainty_factor=0.1, samples=100):
        """
        Initialize SatellitePredictor with TLE data, prediction time settings, and uncertainty parameters.

        Parameters:
        - tle_line1, tle_line2: Two-Line Element (TLE) data for initializing satellite orbital parameters.
        - start_time: The starting time for generating predictions.
        - time_interval: The interval between each position prediction.
        - num_intervals: Total number of prediction intervals.
        - uncertainty_factor: Factor controlling the magnitude of added noise to simulate real-world uncertainties.
        - samples: Number of samples to represent uncertainty in position predictions.
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

        - The SGP4 model uses TLE data for satellite position calculation, factoring in orbital mechanics.
        - TLE-based predictions may not fully capture effects of non-gravitational forces, limiting absolute accuracy.
        
        Returns:
        - positions: Numpy array of satellite positions at each interval.
        - times: List of timestamps corresponding to each predicted position.
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

        - This noise models uncertainties from atmospheric drag, gravitational variations, and other unmodeled factors.
        - The uncertainty factor controls the noise magnitude relative to each position's value.
        
        Returns:
        - A numpy array of position samples with added noise, simulating prediction uncertainties.
        """
        return positions + np.random.normal(0, self.uncertainty_factor * np.abs(positions),
                                            (self.samples, self.num_intervals, 3))

    def preprocess_data(self, positions_with_uncertainty):
        """
        Prepare data for model training by creating features and targets, then normalizing them.

        - Features (X): All but the last position in each sample.
        - Targets (y): All but the first position in each sample, enabling the model to learn sequential movement.
        
        Normalization:
        - Min-Max scaling is used to bring feature and target values into a consistent range, aiding model convergence.
        
        Returns:
        - X, y: Normalized features and targets.
        - X_min, X_max, y_min, y_max: Min-Max normalization factors, needed for reversing normalization in predictions.
        """
        X = positions_with_uncertainty[:, :-1]
        y = positions_with_uncertainty[:, 1:]

        X_min, X_max = np.min(X, axis=(0, 1)), np.max(X, axis=(0, 1))
        y_min, y_max = np.min(y, axis=(0, 1)), np.max(y, axis=(0, 1))

        X = (X - X_min) / (X_max - X_min)
        y = (y - y_min) / (y_max - y_min)

        return X, y, (X_min, X_max), (y_min, y_max)

