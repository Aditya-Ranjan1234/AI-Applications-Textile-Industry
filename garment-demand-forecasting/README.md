# Textile Demand Forecasting

A machine learning-based system for predicting textile demand and optimizing production planning.

## Overview

This project implements advanced demand forecasting models for the textile industry. It helps manufacturers optimize their production planning, reduce waste, and improve resource allocation through accurate demand predictions.

## Features

- Time series forecasting
- Seasonal pattern recognition
- Market trend analysis
- Inventory optimization
- Production planning assistance

## Dataset

The project uses historical data including:
- Sales records
- Seasonal patterns
- Market trends
- Production capacity
- Inventory levels

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
```bash
# Place your data in the data/ directory
```

## Usage

### Training

To train the forecasting model:
```bash
python train.py
```

### Prediction

To generate demand forecasts:
```bash
python predict.py --input data/input.csv --output predictions.csv
```

## Model Architecture

- Multiple forecasting models:
  - ARIMA
  - Prophet
  - LSTM
  - XGBoost
- Ensemble approach for better accuracy
- Feature engineering for seasonal patterns

## Performance

The system provides:
- Accurate demand predictions
- Confidence intervals
- Trend analysis
- Seasonal decomposition

## Directory Structure

```
textile-demand-forecasting/
├── data/               # Dataset directory
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── data/        # Data processing
│   ├── models/      # Model definitions
│   └── utils/       # Utility functions
├── train.py         # Training script
├── predict.py       # Prediction script
└── requirements.txt # Project dependencies
```

## Requirements

- Python 3.8+
- Pandas
- NumPy
- scikit-learn
- Prophet
- XGBoost
- PyTorch (for LSTM)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. 