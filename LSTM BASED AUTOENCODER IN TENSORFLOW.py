import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
data = np.random.rand(1000, 10, 1)  
latent_dim = 2
inputs = Input(shape=(10, 1))
encoded = LSTM(4)(inputs)
encoded = RepeatVector(10)(encoded)
decoded = LSTM(4, return_sequences=True)(encoded)
decoded = TimeDistributed(Dense(1))(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
autoencoder.fit(data, data, epochs=50, batch_size=32, validation_split=0.2)
encoder = Model(inputs, encoded)
encoded_input = Input(shape=(10, 4))
decoder_layer = autoencoder.layers[-2](encoded_input)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = Model(encoded_input, decoder_layer)
encoder.summary()
decoder.summary()
encoded_data = encoder.predict(data[:5])
decoded_data = decoder.predict(encoded_data)
print("Encoded Data:", encoded_data)
print("Decoded Data:", decoded_data)
