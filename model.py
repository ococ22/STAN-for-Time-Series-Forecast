import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Layer, Concatenate, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas import DataFrame, concat
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Model parameters (unchanged)
n_features = 8
Epoch_value = 70
Batch_size_value = 72
learning_rate = 0.001
fc_neurons = 30
drop_out = 0.2
time_steps = 96
output_length = 96
seasonal_period = 288

# Custom Positional Encoding Layer (unchanged)
class PositionalEncoding(Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self._get_positional_encoding(sequence_length, d_model)

    def _get_positional_encoding(self, sequence_length, d_model):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((sequence_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]
        return tf.convert_to_tensor(pe, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
        return config

# Custom Reshape Layer (unchanged)
class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, (-1, *self.target_shape))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.target_shape)

    def get_config(self):
        config = super(ReshapeLayer, self).get_config()
        config.update({"target_shape": self.target_shape})
        return config

# Custom Squeeze Layer (unchanged)
class SqueezeLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SqueezeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super(SqueezeLayer, self).get_config()
        config.update({"axis": self.axis})
        return config

# Convert time series to supervised learning format (unchanged)
def series_to_supervised(data, n_in=time_steps, n_out=output_length, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.iloc[:, 0].shift(-i))  # Only Active_Power
        if i == 0:
            names += ['Active_Power(t)']
        else:
            names += ['Active_Power(t+%d)' % i]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Calculate MAPE in original scale (unchanged)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def STANModel(DataPath, FileName, time_steps, n_features, drop_out, fc_neurons, learning_rate, Epoch_value, Batch_size_value, seasonal_period=288):
    # Load and preprocess data
    dataset = pd.read_csv(DataPath)
    dataset.columns = ['timestamp', 'Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius',
                       'Global_Horizontal_Radiation', 'Wind_Direction', 'Weather_Daily_Rainfall',
                       'Max_Wind_Speed', 'Air_Pressure', 'Hail_Accumulation']
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], dayfirst=True)
    dataset.set_index('timestamp', inplace=True)
    dataset.drop(columns=['Hail_Accumulation'], inplace=True)
    
    # Split data
    n_train = round(dataset.shape[0] * 0.7)
    n_val = round(dataset.shape[0] * 0.2)
    train_df = dataset[:n_train].copy()
    val_df = dataset[n_train:n_train + n_val].copy()
    test_df = dataset[n_train + n_val:].copy()
    
    # Interpolate missing values
    train_df['Active_Power'] = train_df['Active_Power'].interpolate(method='linear').ffill().bfill()
    val_df['Active_Power'] = val_df['Active_Power'].interpolate(method='linear').ffill().bfill()
    test_df['Active_Power'] = test_df['Active_Power'].interpolate(method='linear').ffill().bfill()
    train_df = train_df.interpolate(method='linear').ffill().bfill()
    val_df = val_df.interpolate(method='linear').ffill().bfill()
    test_df = test_df.interpolate(method='linear').ffill().bfill()
    
    # Decompose Active Power
    train_active_power = train_df['Active_Power']
    val_active_power = val_df['Active_Power']
    test_active_power = test_df['Active_Power']
    train_decomposition = seasonal_decompose(train_active_power, model='additive', period=seasonal_period)
    val_decomposition = seasonal_decompose(val_active_power, model='additive', period=seasonal_period)
    test_decomposition = seasonal_decompose(test_active_power, model='additive', period=seasonal_period)
    
    train_df['Trend'] = train_decomposition.trend
    train_df['Seasonal'] = train_decomposition.seasonal
    val_df['Trend'] = val_decomposition.trend
    val_df['Seasonal'] = val_decomposition.seasonal
    test_df['Trend'] = test_decomposition.trend
    test_df['Seasonal'] = test_decomposition.seasonal
    train_df[['Trend', 'Seasonal']] = train_df[['Trend', 'Seasonal']].interpolate(method='linear').ffill().bfill()
    val_df[['Trend', 'Seasonal']] = val_df[['Trend', 'Seasonal']].interpolate(method='linear').ffill().bfill()
    test_df[['Trend', 'Seasonal']] = test_df[['Trend', 'Seasonal']].interpolate(method='linear').ffill().bfill()
    
    # Convert to numpy arrays
    columns = ['Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius',
               'Global_Horizontal_Radiation', 'Wind_Direction', 'Weather_Daily_Rainfall',
               'Max_Wind_Speed', 'Air_Pressure', 'Trend', 'Seasonal']
    train = train_df[columns].values.astype('float32')
    val = val_df[columns].values.astype('float32')
    test = test_df[columns].values.astype('float32')

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    # Convert to supervised learning format
    train_reframed = series_to_supervised(train_scaled, time_steps, output_length)
    val_reframed = series_to_supervised(val_scaled, time_steps, output_length)
    test_reframed = series_to_supervised(test_scaled, time_steps, output_length)

    train_values = train_reframed.values
    val_values = val_reframed.values
    test_values = test_reframed.values

    # Prepare inputs and outputs
    n_obs = time_steps * (n_features + 2)
    train_X = train_values[:, :n_obs]
    train_y = train_values[:, -output_length:]
    val_X = val_values[:, :n_obs]
    val_y = val_values[:, -output_length:]
    test_X = test_values[:, :n_obs]
    test_y = test_values[:, -output_length:]
    train_X = train_X.reshape((train_X.shape[0], time_steps, n_features + 2))
    val_X = val_X.reshape((val_X.shape[0], time_steps, n_features + 2))
    test_X = test_X.reshape((test_X.shape[0], time_steps, n_features + 2))

    # Prepare target sequence for decoder (shifted input for teacher forcing)
    train_y_shifted = np.zeros_like(train_y)
    val_y_shifted = np.zeros_like(val_y)
    test_y_shifted = np.zeros_like(test_y)
    train_y_shifted[:, 1:] = train_y[:, :-1]
    val_y_shifted[:, 1:] = val_y[:, :-1]
    test_y_shifted[:, 1:] = test_y[:, :-1]
    train_y_shifted = train_y_shifted.reshape((train_y_shifted.shape[0], output_length, 1))
    val_y_shifted = val_y_shifted.reshape((val_y_shifted.shape[0], output_length, 1))
    test_y_shifted = test_y_shifted.reshape((test_y_shifted.shape[0], output_length, 1))

    # Build model with MLP-based decoder
    # Encoder
    x = Input(shape=(time_steps, n_features + 2))
    trend_input = x[:, :, -2]
    seasonal_input = x[:, :, -1]
    exogenous_input = x[:, :, :-2]
    reshape_input = ReshapeLayer(target_shape=(time_steps, 1))
    trend_input = reshape_input(trend_input)
    seasonal_input = reshape_input(seasonal_input)
    pos_encoding = PositionalEncoding(sequence_length=time_steps, d_model=1)
    trend_encoded = pos_encoding(trend_input)
    seasonal_encoded = pos_encoding(seasonal_input)
    trend_flat = Flatten()(trend_encoded)
    trend_dense = Dense(fc_neurons, activation='relu')(trend_flat)
    seasonal_flat = Flatten()(seasonal_encoded)
    seasonal_dense = Dense(fc_neurons, activation='relu')(seasonal_flat)
    composite_feature = trend_dense + seasonal_dense
    composite_dense = Dense(fc_neurons, activation='relu')(composite_feature)
    exogenous_flat = Flatten()(exogenous_input)
    exogenous_dense = Dense(fc_neurons, activation='relu')(exogenous_flat)
    encoder_output = Concatenate()([trend_dense, seasonal_dense, composite_dense, exogenous_dense])
    encoder_output = Dense(fc_neurons, activation='relu')(encoder_output)
    encoder_output = ReshapeLayer(target_shape=(1, fc_neurons))(encoder_output)

    # Decoder input
    decoder_input = Input(shape=(output_length, 1))
    decoder_pos_encoding = PositionalEncoding(sequence_length=output_length, d_model=1)(decoder_input)
    decoder_flat = Flatten()(decoder_pos_encoding)
    
    # MLP Decoder
    # MLP Decoder
    combined_input = Concatenate()([decoder_flat, Flatten()(encoder_output)])
    mlp_layer1 = Dense(fc_neurons * 4, activation='relu')(combined_input)
    mlp_output = Dense(output_length)(mlp_layer1)
    outputs = ReshapeLayer(target_shape=(output_length, 1))(mlp_output)
    outputs = SqueezeLayer(axis=-1)(outputs)  # Shape: (batch_size, output_length)

    # Define model
    model = Model(inputs=[x, decoder_input], outputs=outputs)
    model.compile(loss='mae', optimizer=Adam(learning_rate=learning_rate))

    # Train model
    history = model.fit(
        [train_X, train_y_shifted], train_y,
        epochs=Epoch_value,
        batch_size=Batch_size_value,
        validation_data=([val_X, val_y_shifted], val_y),
        verbose=0,
        shuffle=False
    )

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

# Plotting functions (unchanged)
def plot_loss(history, run_number):
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
