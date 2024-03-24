from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

embedding_size=32
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 64
num_epochs = 1
X_valid, y_valid = xtrain[:batch_size], ytrain[:batch_size]
X_train2, y_train2 = xtrain[batch_size:], ytrain[batch_size:]

model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=500, epochs=num_epochs)

scores = model.evaluate(xtest, ytest, verbose=0)
print('Test accuracy:', scores[1])
