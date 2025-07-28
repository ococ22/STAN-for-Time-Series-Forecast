    # Evaluate model
    yhat = model.predict([test_X, test_y_shifted], verbose=0)
    test_y_full = np.zeros((test_y.shape[0], output_length, n_features + 2))
    yhat_full = np.zeros((yhat.shape[0], output_length, n_features + 2))
    test_y_full[:, :, 0] = test_y
    yhat_full[:, :, 0] = yhat
    test_y_original = np.zeros((test_y.shape[0], output_length))
    yhat_original = np.zeros((yhat.shape[0], output_length))
    for i in range(test_y.shape[0]):
        test_y_original[i, :] = scaler.inverse_transform(test_y_full[i, :, :])[:, 0]
        yhat_original[i, :] = scaler.inverse_transform(yhat_full[i, :, :])[:, 0]
    test_y_flat = test_y.flatten()
    yhat_flat = yhat.flatten()
    rmse = np.sqrt(mean_squared_error(test_y_flat, yhat_flat))
    mae = mean_absolute_error(test_y_flat, yhat_flat)
    mape = mean_absolute_percentage_error(test_y_original.flatten(), yhat_original.flatten())
    r_squared = r2_score(test_y_flat, yhat_flat)

    return rmse, mae, mape, r_squared, model, history, test_y_original, yhat_original
