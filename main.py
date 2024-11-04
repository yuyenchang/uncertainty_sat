# main.py

from datetime import datetime, timedelta
from satellite_predictor import SatellitePredictor
from rnn_model import RNNModel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_predictions(prediction_mean, prediction_uncertainty):
    """
    Visualize predictions and uncertainties in different 3D views to enhance understanding.
    Displays the mean predictions along with uncertainty bands from multiple perspectives.
    """
    time_steps = np.arange(prediction_mean.shape[1]) * 10  # Time steps in minutes
    fig = plt.figure(figsize=(12, 12))
    angles = [(10, 60), (30, 120), (50, 150), (70, 240)]  # Different elevation and azimuth angles for 3D views
    titles = ['View 1: Elevation 10, Azimuth 60',
              'View 2: Elevation 30, Azimuth 120',
              'View 3: Elevation 50, Azimuth 150',
              'View 4: Elevation 70, Azimuth 240']
    
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.plot(*prediction_mean[0].T, label="Mean Prediction")  # Plot mean prediction over time

        # Adding uncertainty bands (random noise) around each point in the mean prediction
        for i in range(len(prediction_mean[0])):
            x_uncertainty = np.random.normal(0, prediction_uncertainty[0, i, 0])
            y_uncertainty = np.random.normal(0, prediction_uncertainty[0, i, 1])
            z_uncertainty = np.random.normal(0, prediction_uncertainty[0, i, 2])

            # Display uncertainty by plotting bands around each mean prediction point
            ax.plot([prediction_mean[0, i, 0] - x_uncertainty, prediction_mean[0, i, 0] + x_uncertainty],
                    [prediction_mean[0, i, 1] - y_uncertainty, prediction_mean[0, i, 1] + y_uncertainty],
                    [prediction_mean[0, i, 2] - z_uncertainty, prediction_mean[0, i, 2] + z_uncertainty],
                    color="gray", alpha=0.3, label="Uncertainty" if i == 0 else "")
        
        # Label axes and set specific view angles for each subplot
        ax.set_xlabel("X Position (km)")
        ax.set_ylabel("Y Position (km)")
        ax.set_zlabel("Z Position (km)")
        ax.set_title(titles[idx])
        ax.view_init(elev=elev, azim=azim)
        ax.legend()
    plt.tight_layout()
    plt.savefig("uncertainty_sat.png")  # Save plot to file
    # plt.show()  # Uncomment to display plot in a window

def main():
    """
    Main function to set up data, train a model, and visualize predictions with uncertainty.
    
    This includes initializing TLE data for satellite position prediction, processing data
    with uncertainty, training an RNN model for sequential prediction, and visualizing the results.
    """
    # Two-Line Element (TLE) data contains essential satellite orbital parameters
    # for accurate positioning in space. Choice of TLE data ensures reliability in satellite tracking.
    tle_line1 = "1 25544U 98067A   20335.54791667  .00001264  00000-0  29623-4 0  9991"
    tle_line2 = "2 25544  51.6441  21.0125 0001399  92.4587 267.6706 15.49346029257441"
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_interval = timedelta(minutes=10)  # 10-minute interval to capture short-term movements

    # Set up SatellitePredictor to generate data with SGP4 model (assumes minimal drag and stable orbit).
    predictor = SatellitePredictor(tle_line1, tle_line2, start_time, time_interval, num_intervals=1000)
    positions, _ = predictor.generate_orbital_data()
    positions_with_uncertainty = predictor.add_uncertainty(positions)  # Adding noise to model uncertainties in real-world scenarios
    X, y, (X_min, X_max), (y_min, y_max) = predictor.preprocess_data(positions_with_uncertainty)  # Normalize data

    # Set up and train an RNN model to learn from sequential satellite data
    # Choice of model architecture (LSTM layers) is critical for capturing time dependencies in orbital data.
    # LSTMs are particularly suited for such data as they retain important information across time steps.
    rnn_model = RNNModel(input_shape=(X.shape[1], X.shape[2]))
    rnn_model.train(X, y)

    # Generate predictions with uncertainty estimation
    # Model employs dropout to estimate uncertainty by sampling multiple predictions, enhancing prediction reliability.
    sample_input = X[:1]  # Take a sample from data for prediction
    prediction_mean, prediction_uncertainty = rnn_model.predict_with_uncertainty(sample_input, X_min=X_min, X_max=X_max)
    plot_predictions(prediction_mean, prediction_uncertainty)

# Run the main function if the script is executed
if __name__ == '__main__':
    main()
