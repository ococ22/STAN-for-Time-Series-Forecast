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
