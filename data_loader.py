import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def series_to_supervised(data, n_in, n_out, dropnan=True):
    """
    Convert time series data into supervised learning format.
    Inputs:
        data: numpy array or DataFrame
        n_in: number of past time steps (input window)
        n_out: number of future time steps to predict (output window)
        dropnan: whether to drop rows with NaN values
    Returns:
        DataFrame of supervised data
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n_out-1)
    for i in range(n_out):
        cols.append(df.iloc[:, 0].shift(-i))  # Assuming target is first variable
        if i == 0:
            names += ['Active_Power(t)']
        else:
            names += ['Active_Power(t+%d)' % i]
    
    agg = concat(cols, axis=1)
    agg.columns = names
    
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg

def load_and_preprocess_data(filepath, time_steps, output_length, seasonal_period=288):
    """
    Load CSV, preprocess data, decompose time series, scale, and prepare supervised data.
    Returns train, val, test sets and scaler.
    """
    # Load data
    dataset = pd.read_csv(filepath)
    dataset.columns = ['timestamp', 'Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius',
                       'Global_Horizontal_Radiation', 'Wind_Direction', 'Weather_Daily_Rainfall',
                       'Max_Wind_Speed', 'Air_Pressure', 'Hail_Accumulation']
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], dayfirst=True)
    dataset.set_index('timestamp', inplace=True)
    dataset.drop(columns=['Hail_Accumulation'], inplace=True)

    # Split into train, val, test
    n_train = round(dataset.shape[0] * 0.7)
    n_val = round(dataset.shape[0] * 0.2)
    train_df = dataset[:n_train].copy()
    val_df = dataset[n_train:n_train + n_val].copy()
    test_df = dataset[n_train + n_val:].copy()

    # Interpolate missing values
    for df in [train_df, val_df, test_df]:
        df.interpolate(method='linear', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    # Decompose Active Power time series
    for df in [train_df, val_df, test_df]:
        decomposition = seasonal_decompose(df['Active_Power'], model='additive', period=seasonal_period)
        df['Trend'] = decomposition.trend
        df['Seasonal'] = decomposition.seasonal
        df[['Trend', 'Seasonal']] = df[['Trend', 'Seasonal']].interpolate(method='linear').ffill().bfill()

    # Columns used for modeling
    columns = ['Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius',
               'Global_Horizontal_Radiation', 'Wind_Direction', 'Weather_Daily_Rainfall',
               'Max_Wind_Speed', 'Air_Pressure', 'Trend', 'Seasonal']

    # Convert to numpy arrays
    train = train_df[columns].values.astype('float32')
    val = val_df[columns].values.astype('float32')
    test = test_df[columns].values.astype('float32')

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    # Prepare supervised datasets
    train_supervised = series_to_supervised(train_scaled, time_steps, output_length)
    val_supervised = series_to_supervised(val_scaled, time_steps, output_length)
    test_supervised = series_to_supervised(test_scaled, time_steps, output_length)

    # Extract values and reshape inputs and outputs
    n_features = train.shape[1]
    n_obs = time_steps * n_features

    def prepare_xy(data_supervised):
        values = data_supervised.values
        X = values[:, :n_obs]
        y = values[:, -output_length:]
        X = X.reshape((X.shape[0], time_steps, n_features))
        return X, y

    train_X, train_y = prepare_xy(train_supervised)
    val_X, val_y = prepare_xy(val_supervised)
    test_X, test_y = prepare_xy(test_supervised)

    # Prepare shifted decoder inputs (teacher forcing)
    def create_shifted_y(y):
        y_shifted = np.zeros_like(y)
        y_shifted[:, 1:] = y[:, :-1]
        y_shifted = y_shifted.reshape((y_shifted.shape[0], output_length, 1))
        return y_shifted

    train_y_shifted = create_shifted_y(train_y)
    val_y_shifted = create_shifted_y(val_y)
    test_y_shifted = create_shifted_y(test_y)

    return (train_X, train_y, train_y_shifted,
            val_X, val_y, val_y_shifted,
            test_X, test_y, test_y_shifted,
            scaler, n_features)
