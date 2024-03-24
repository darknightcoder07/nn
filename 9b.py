from keras.preprocessing import sequence
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()
max_words = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)
print(x_train[2])
