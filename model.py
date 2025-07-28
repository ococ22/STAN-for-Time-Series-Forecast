from positional_encoding import PositionalEncoding
from reshape_layer import ReshapeLayer
from squeeze_layer import SqueezeLayer

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Layer, Concatenate
from keras.optimizers import Adam

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Concatenate
from keras.optimizers import Adam

from positional_encoding import PositionalEncoding
from reshape_layer import ReshapeLayer
from squeeze_layer import SqueezeLayer

def build_stan_model(time_steps, n_features, fc_neurons, drop_out, output_length, learning_rate):
    # Encoder input
    x = Input(shape=(time_steps, n_features + 2), name='encoder_input')

    # Split trend, seasonal, exogenous from input
    trend_input = x[:, :, -2]
    seasonal_input = x[:, :, -1]
    exogenous_input = x[:, :, :-2]

    # Reshape trend and seasonal to (batch_size, time_steps, 1)
    reshape_layer = ReshapeLayer(target_shape=(time_steps, 1))
    trend_reshaped = reshape_layer(trend_input)
    seasonal_reshaped = reshape_layer(seasonal_input)

    # Positional encoding on trend and seasonal
    pos_encoding = PositionalEncoding(sequence_length=time_steps, d_model=1)
    trend_encoded = pos_encoding(trend_reshaped)
    seasonal_encoded = pos_encoding(seasonal_reshaped)

    # Flatten and dense layers on trend and seasonal
    trend_flat = Flatten()(trend_encoded)
    trend_dense = Dense(fc_neurons, activation='relu')(trend_flat)

    seasonal_flat = Flatten()(seasonal_encoded)
    seasonal_dense = Dense(fc_neurons, activation='relu')(seasonal_flat)

    # Composite feature from trend and seasonal dense layers
    composite_feature = trend_dense + seasonal_dense
    composite_dense = Dense(fc_neurons, activation='relu')(composite_feature)

    # Exogenous dense layer
    exogenous_flat = Flatten()(exogenous_input)
    exogenous_dense = Dense(fc_neurons, activation='relu')(exogenous_flat)

    # Concatenate all encoder features
    encoder_concat = Concatenate()([trend_dense, seasonal_dense, composite_dense, exogenous_dense])
    encoder_dense = Dense(fc_neurons, activation='relu')(encoder_concat)

    # Reshape encoder output to (batch_size, 1, fc_neurons)
    encoder_output = ReshapeLayer(target_shape=(1, fc_neurons))(encoder_dense)

    # Decoder input
    decoder_input = Input(shape=(output_length, 1), name='decoder_input')
    decoder_pos_encoded = PositionalEncoding(sequence_length=output_length, d_model=1)(decoder_input)

    decoder_flat = Flatten()(decoder_pos_encoded)

    # Concatenate decoder flat and encoder output (flattened)
    combined_input = Concatenate()([decoder_flat, Flatten()(encoder_output)])

    # MLP decoder layers
    mlp_1 = Dense(fc_neurons * 4, activation='relu')(combined_input)
    mlp_output = Dense(output_length)(mlp_1)

    # Reshape to (batch_size, output_length, 1) and squeeze last dim
    output_reshaped = ReshapeLayer(target_shape=(output_length, 1))(mlp_output)
    outputs = SqueezeLayer(axis=-1)(output_reshaped)  # Final shape: (batch_size, output_length)

    # Define and compile model
    model = Model(inputs=[x, decoder_input], outputs=outputs, name='STAN_Model')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mae')

    return model
