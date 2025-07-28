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
