# main.py

from datetime import datetime, timedelta
from satellite_predictor import SatellitePredictor
from rnn_model import RNNModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_predictions(prediction_mean, prediction_uncertainty):
    # Visualize predictions and uncertainties in different 3D views
    time_steps = np.arange(prediction_mean.shape[1]) * 10  # Time steps in minutes
    fig = plt.figure(figsize=(12, 12))
    angles = [(10, 60), (30, 120), (50, 150), (70, 240)]
    titles = ['View 1: Elevation 10, Azimuth 60',
              'View 2: Elevation 30, Azimuth 120',
              'View 3: Elevation 50, Azimuth 150',
              'View 4: Elevation 70, Azimuth 240']
    
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.plot(*prediction_mean[0].T, label="Mean Prediction")
        for i in range(len(prediction_mean[0])):
            x_uncertainty = np.random.normal(0, prediction_uncertainty[0, i, 0])
            y_uncertainty = np.random.normal(0, prediction_uncertainty[0, i, 1])
            z_uncertainty = np.random.normal(0, prediction_uncertainty[0, i, 2])
            ax.plot([prediction_mean[0, i, 0] - x_uncertainty, 
                     prediction_mean[0, i, 0] + x_uncertainty],
                    [prediction_mean[0, i, 1] - y_uncertainty, 
                     prediction_mean[0, i, 1] + y_uncertainty],
                    [prediction_mean[0, i, 2] - z_uncertainty, 
                     prediction_mean[0, i, 2] + z_uncertainty], color="gray", alpha=0.3, label="Uncertainty" if i == 0 else "")
        
        ax.set_xlabel("X Position (km)")
        ax.set_ylabel("Y Position (km)")
        ax.set_zlabel("Z Position (km)")
        ax.set_title(titles[idx])
        ax.view_init(elev=elev, azim=azim)
        ax.legend()
    plt.tight_layout()
    plt.savefig("uncertainty_iss.png")
    plt.show()

def main():
    # Initialize TLE data and prediction parameters
    tle_line1 = "1 25544U 98067A   20335.54791667  .00001264  00000-0  29623-4 0  9991"
    tle_line2 = "2 25544  51.6441  21.0125 0001399  92.4587 267.6706 15.49346029257441"
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_interval = timedelta(minutes=10)

    # Create a SatellitePredictor instance and generate data
    predictor = SatellitePredictor(tle_line1, tle_line2, start_time, time_interval, num_intervals=1000)
    positions, _ = predictor.generate_orbital_data()
    positions_with_uncertainty = predictor.add_uncertainty(positions)
    X, y, (X_min, X_max), (y_min, y_max) = predictor.preprocess_data(positions_with_uncertainty)

    # Train the RNN model
    rnn_model = RNNModel(input_shape=(X.shape[1], X.shape[2]))
    rnn_model.train(X, y)

    # Make predictions and visualize
    sample_input = X[:1]
    prediction_mean, prediction_uncertainty = rnn_model.predict_with_uncertainty(sample_input, X_min=X_min, X_max=X_max)
    plot_predictions(prediction_mean, prediction_uncertainty)

# Run the main function if the script is executed
if __name__ == '__main__':
    main()
