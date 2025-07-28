import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10  # to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def evaluate_model(model, test_X, test_y, test_y_shifted, scaler, n_features, output_length):
    # Predict using model
    yhat = model.predict([test_X, test_y_shifted], verbose=0)
    
    # Prepare arrays to inverse transform (adding dummy features for scaler)
    test_y_full = np.zeros((test_y.shape[0], output_length, n_features + 2))
    yhat_full = np.zeros((yhat.shape[0], output_length, n_features + 2))
    
    # Put predictions and true values in first feature column, rest are zeros
    test_y_full[:, :, 0] = test_y
    yhat_full[:, :, 0] = yhat
    
    # Inverse transform to original scale feature-wise
    test_y_original = np.zeros((test_y.shape[0], output_length))
    yhat_original = np.zeros((yhat.shape[0], output_length))
    for i in range(test_y.shape[0]):
        test_y_original[i, :] = scaler.inverse_transform(test_y_full[i, :, :])[:, 0]
        yhat_original[i, :] = scaler.inverse_transform(yhat_full[i, :, :])[:, 0]
    
    # Flatten for metrics that do not require original scale
    test_y_flat = test_y.flatten()
    yhat_flat = yhat.flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_y_flat, yhat_flat))
    mae = mean_absolute_error(test_y_flat, yhat_flat)
    mape = mean_absolute_percentage_error(test_y_original.flatten(), yhat_original.flatten())
    r_squared = r2_score(test_y_flat, yhat_flat)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r_squared": r_squared,
        "test_y_original": test_y_original,
        "yhat_original": yhat_original
    }
