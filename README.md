# Uncertainty Quantification in Satellite Orbits

This project provides a model to predict the orbital position of a satellite, specifically the International Space Station (ISS), using Two-Line Element (TLE) data. It uses a Recurrent Neural Network (RNN) with LSTM layers to predict future satellite positions and includes uncertainty estimation to account for prediction uncertainties.

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

Notes:
- Python 3.9.1 is recommended.
- Use a virtual environment to avoid conflicts.

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

This project includes the following method for uncertainty estimation:

- **Monte Carlo (MC) Dropout**

## Results

The project generates 3D visualizations of the satellite's orbit with uncertainty bands, highlighting positional uncertainty and model reliability.

<img src="https://github.com/yuyenchang/uncertainty_sat/blob/main/uncertainty_sat.png" alt="Example Image" style="width:80%;"/>

## License

This project is open-source and available under the MIT License.
