# Satellite Orbit Prediction with Uncertainty
This project provides a model to predict the orbital position of a satellite, specifically the International Space Station (ISS), using Two-Line Element (TLE) data. It uses an RNN with LSTM layers to predict the satellite's future positions and includes uncertainty estimation to account for prediction variations.

Features
Loads and uses TLE data for satellite position predictions.
Adds uncertainty through Gaussian noise for more realistic predictions.
RNN-based model with LSTM layers to capture time dependencies in orbital data.
Visualizations of predicted orbit path with uncertainty bands.
Requirements
Make sure to install the required packages:

bash
Copy code
pip install numpy sgp4 tensorflow matplotlib
Usage
1. Initialize Satellite Predictor and Generate Orbital Data
The SatellitePredictor class initializes the satellite using TLE data and generates orbital data over a specified time interval.

python
Copy code
from datetime import datetime, timedelta

tle_line1 = "1 25544U 98067A   20335.54791667  .00001264  00000-0  29623-4 0  9991"
tle_line2 = "2 25544  51.6441  21.0125 0001399  92.4587 267.6706 15.49346029257441"
start_time = datetime(2024, 1, 1, 0, 0, 0)  # Set the starting time for the simulation
time_interval = timedelta(minutes=10)  # Define the interval for predictions

predictor = SatellitePredictor(tle_line1, tle_line2, start_time, time_interval, num_intervals=1000)
positions, _ = predictor.generate_orbital_data()
positions_with_uncertainty = predictor.add_uncertainty(positions)
X, y, (X_min, X_max), (y_min, y_max) = predictor.preprocess_data(positions_with_uncertainty)
2. Train the RNN Model
The RNNModel class creates and trains an RNN model to predict the satellite's position. You can modify the epochs and batch_size parameters as needed.

python
Copy code
rnn_model = RNNModel(input_shape=(X.shape[1], X.shape[2]))
rnn_model.train(X, y, epochs=100, batch_size=32)
3. Make Predictions with Uncertainty
Use the model to make predictions with uncertainty. The predict_with_uncertainty function generates predictions and calculates the uncertainty by enabling dropout during inference.

python
Copy code
sample_input = X[:1]
prediction_mean, prediction_uncertainty = rnn_model.predict_with_uncertainty(sample_input, X_min=X_min, X_max=X_max)
4. Visualize Predictions and Uncertainty
The predictions and uncertainty are visualized in 3D plots from different angles, saved as uncertainty_iss.png.

python
Copy code
# Import matplotlib for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

time_steps = np.arange(prediction_mean.shape[1]) * 10  # Time steps in minutes

# Create different views of the predictions
fig = plt.figure(figsize=(12, 12))
angles = [(10, 60), (30, 120), (50, 150), (70, 240)]
titles = ['View 1: Elevation 10, Azimuth 60',
          'View 2: Elevation 30, Azimuth 120',
          'View 3: Elevation 50, Azimuth 150',
          'View 4: Elevation 70, Azimuth 240']
for idx, (elev, azim) in enumerate(angles):
    ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
    ax.plot(*prediction_mean[0].T, label="Mean Prediction")
    ax.set_xlabel("X Position (km)")
    ax.set_ylabel("Y Position (km)")
    ax.set_zlabel("Z Position (km)")
    ax.set_title(titles[idx])
    ax.view_init(elev=elev, azim=azim)
    ax.legend()
plt.tight_layout()
plt.savefig("uncertainty_iss.png")

# Display 3D plot of the predicted orbit
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(*prediction_mean[0].T, label="Mean Prediction")
ax.set_xlabel("X Position (km)")
ax.set_ylabel("Y Position (km)")
ax.set_zlabel("Z Position (km)")
ax.set_title("3D Satellite Orbit Prediction with Uncertainty")
ax.legend()
plt.show()
File Structure
SatellitePredictor: A class that loads TLE data, generates orbital data, and adds uncertainty.
RNNModel: A class that trains and makes predictions with an RNN model.
uncertainty_iss.png: The generated file with 3D visualizations of the orbit and uncertainties.
Example Output
After running the code, you will get 3D plots that show the satellite's predicted orbit and highlight uncertainty bands, providing insights into prediction reliability.

License
This project is open-source under the MIT License.
