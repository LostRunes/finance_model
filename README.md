# Electric Load Forecasting Model

This repository contains a robust electric load forecasting pipeline built with Python and Scikit-learn.

## Features
- **Data Integration**: Merges load data with external weather factors and holiday events.
- **Feature Engineering**: Includes 24h/7d lags and rolling averages.
- **Cost-Aware Optimization**: Uses a custom asymmetric penalty function (4x underforecast, 2x overforecast).
- **Multiplier Sweep**: Automatically finds the optimal forecast adjustment to minimize total penalty.
- **Baseline Comparison**: Compares model performance against a naive lag-based baseline.

## Installation
```bash
pip install pandas scikit-learn numpy
```

## Usage
Simply run the main script:
```bash
python load_forecasting.py
```

## Performance
The model consistently outperforms the naive baseline (Lag-96) by significant margins in terms of total penalty.
