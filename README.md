# Uncertainty Quantification for Satellite Orbit

This project provides a model to predict the orbital position of a satellite, specifically the International Space Station (ISS), using Two-Line Element (TLE) data. It uses a Recurrent Neural Network (RNN) with LSTM layers to predict future satellite positions and includes uncertainty estimation to account for prediction variations.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Uncertainty Estimation Methods](#uncertainty-estimation-methods)
- [Results](#results)
- [License](#license)

## Project Overview

The Satellite Orbit Prediction project calculates the orbital position of a satellite over a specified period using TLE data. This project implements an RNN model that leverages LSTM layers for satellite position prediction, with added uncertainty quantification through dropout to simulate variations in the satellite’s trajectory.

## Features

- **Satellite Prediction**: Loads TLE data for ISS and generates satellite position predictions.
- **Uncertainty Estimation**: Adds random Gaussian noise to simulate uncertainty in predictions.
- **RNN Model with LSTM**: A neural network model that accounts for temporal dependencies in satellite position data.
- **Visualization**: Plots 3D trajectory predictions with uncertainty bands for various viewpoints.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/uncertainty_sat.git
    cd uncertainty_sat
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    Ensure `requirements.txt` includes necessary packages such as `numpy`, `tensorflow`, `matplotlib`, and `sgp4`.

## Usage

1. **Initialize and Configure Parameters**:
   
   Update TLE data, starting time, and time interval in `main.py`.

2. **Run the Script**:

    ```bash
    python main.py
    ```

    The script will generate satellite positions, train an RNN model, and visualize predictions with uncertainty.

## Directory Structure

```graphql
uncertainty_sat/
├── main.py                   # Main script to run the entire pipeline
├── satellite_predictor.py    # Handles satellite data generation and preprocessing
├── rnn_model.py              # Defines the RNN model architecture
├── requirements.txt          # List of required packages
└── uncertainty_sat.png       # Example output plot showing orbit with uncertainty
```

## Uncertainty Estimation Methods

The project includes uncertainty quantification through:

- **Monte Carlo Dropout**: Enables dropout during inference to estimate uncertainty by generating multiple predictions.

## Results

The project generates 3D visualizations of the satellite's predicted orbit with uncertainty bands. Different views highlight positional uncertainty, providing insights into model reliability over time.

## License

This project is open-source and available under the MIT License.
