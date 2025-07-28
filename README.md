# STAN Time Series Forecasting Model

This repository contains a TensorFlow/Keras implementation of the Seasonality-Trend Aware Network (STAN) model for multivariate time series forecasting. It incorporates custom layers and MLP-based decoder architecture to predict future values of active power with seasonal decomposition.

---

## Project Structure

- `data_loader.py`  
  Data loading, preprocessing, normalization, and preparation of input/output sequences.

- `layers.py`  
  Custom Keras layers: `PositionalEncoding`, `ReshapeLayer`, and `SqueezeLayer`.

- `model.py`  
  Defines the STAN model architecture and returns the compiled model.

- `evaluate.py`  
  Functions to evaluate the model’s predictions, including inverse scaling and metric calculations.

- `plotting.py`  
  Functions to plot training loss and actual vs predicted results.

- `run_training.py` (optional)  
  Script to orchestrate loading data, building the model, training, evaluation, and plotting.

---

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- numpy  
- pandas  
- scikit-learn  
- matplotlib  
- statsmodels  

You can install dependencies using:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib statsmodels
