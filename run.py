from data_loader import load_and_preprocess_data
from model import build_STAN_model
from evaluate import evaluate_model
from plotting import plot_loss, plot_actual_vs_predicted

# Set parameters
data_path = 'your_data.csv'
file_name = 'your_data.csv'  # or however you want to handle this
time_steps = 96
n_features = 8
drop_out = 0.2
fc_neurons = 30
learning_rate = 0.001
Epoch_value = 70
Batch_size_value = 72
seasonal_period = 288

# Load and preprocess data
train_X, train_y, val_X, val_y, test_X, test_y, train_y_shifted, val_y_shifted, test_y_shifted, scaler = load_and_preprocess_data(
    data_path,
    time_steps,
    n_features,
    seasonal_period=seasonal_period
)

# Build the model
model = build_STAN_model(time_steps, n_features, drop_out, fc_neurons, learning_rate, output_length=96)

# Train the model
history = model.fit(
    [train_X, train_y_shifted], train_y,
    epochs=Epoch_value,
    batch_size=Batch_size_value,
    validation_data=([val_X, val_y_shifted], val_y),
    verbose=1,
    shuffle=False
)

# Evaluate
rmse, mae, mape, r_squared, test_y_original, yhat_original = evaluate_model(
    model, test_X, test_y, test_y_shifted, scaler, n_features, output_length=96
)

print(f'RMSE: {rmse}, MAE: {mae}, MAPE: {mape}, R^2: {r_squared}')

# Plot results
plot_loss(history, run_number=1)
plot_actual_vs_predicted(test_y_original, yhat_original, run_number=1)
