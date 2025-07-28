import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

class DataLoader:
    def __init__(self, data_path, time_steps, output_length, n_features, seasonal_period=288):
        self.data_path = data_path
        self.time_steps = time_steps
        self.output_length = output_length
        self.n_features = n_features
        self.seasonal_period = seasonal_period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def load_and_prepare(self):
        dataset = pd.read_csv(self.data_path)
        dataset.columns = ['timestamp', 'Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius',
                           'Global_Horizontal_Radiation', 'Wind_Direction', 'Weather_Daily_Rainfall',
                           'Max_Wind_Speed', 'Air_Pressure', 'Hail_Accumulation']
        dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], dayfirst=True)
        dataset.set_index('timestamp', inplace=True)
        dataset.drop(columns=['Hail_Accumulation'], inplace=True)

        # Train/val/test split
        n_train = round(dataset.shape[0] * 0.7)
        n_val = round(dataset.shape[0] * 0.2)
        train_df = dataset[:n_train].copy()
        val_df = dataset[n_train:n_train + n_val].copy()
        test_df = dataset[n_train + n_val:].copy()

        # Fill missing values
        for df in [train_df, val_df, test_df]:
            df.interpolate(method='linear', inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)

        # Seasonal decomposition
        for df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
            decomposition = seasonal_decompose(df['Active_Power'], model='additive', period=self.seasonal_period)
            df['Trend'] = decomposition.trend
            df['Seasonal'] = decomposition.seasonal
            df[['Trend', 'Seasonal']] = df[['Trend', 'Seasonal']].interpolate(method='linear').ffill().bfill()

        # Extract and scale features
        columns = ['Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius',
                   'Global_Horizontal_Radiation', 'Wind_Direction', 'Weather_Daily_Rainfall',
                   'Max_Wind_Speed', 'Air_Pressure', 'Trend', 'Seasonal']
        
        train = train_df[columns].values.astype('float32')
        val = val_df[columns].values.astype('float32')
        test = test_df[columns].values.astype('float32')

        self.scaler.fit(train)
        train_scaled = self.scaler.transform(train)
        val_scaled = self.scaler.transform(val)
        test_scaled = self.scaler.transform(test)

        return train_scaled, val_scaled, test_scaled, self.scaler
