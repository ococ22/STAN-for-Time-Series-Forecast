import numpy as np
import matplotlib.pyplot as plt

def plot_loss(history, run_number):
    """
    Plot training and validation loss (MAE) over epochs.

    Args:
        history: Keras History object returned by model.fit().
        run_number: Integer or string identifier for the run.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss (MAE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MAE)')
    plt.title(f'Training and Validation Loss - Run {run_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'loss_plot_run_{run_number}.png')
    plt.show()
    plt.close()

def plot_actual_vs_predicted(test_y, yhat, run_number):
    """
    Plot actual vs predicted Active Power for the first timestep of each sample.

    Args:
        test_y: Ground truth values, shape (samples, output_length).
        yhat: Predicted values, same shape as test_y.
        run_number: Integer or string identifier for the run.
    """
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(test_y.shape[0])
    plt.plot(time_steps, test_y[:, 0], label='Actual Active Power', color='blue')
    plt.plot(time_steps, yhat[:, 0], label='Predicted Active Power', color='orange')
    plt.title(f'Actual vs Predicted Active Power (First Step) - Run {run_number}')
    plt.xlabel('Sample')
    plt.ylabel('Active Power (Original Scale)')
    plt.legend()
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'actual_vs_predicted_run_{run_number}.png')
    plt.show()
    plt.close()
