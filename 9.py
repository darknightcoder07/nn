from keras.datasets import imdb
vocabulary_size = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabulary_size)
print(f"Dataset with {len(x_train)} training samples and {len(x_test)} test samples")
print("Preview of dataset:")
print(x_train[2])
print(y_train[2])
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---Review with words---')
print([id2word.get(i, ' ') for i in x_train[2]])
print('---Label---')
print(y_train[2])
max_review_length = max((len(x) for x in x_train + x_test))
min_review_length = min((len(x) for x in x_train + x_test))
print(f"Max review length: {max_review_length}")
print(f"Min review length: {min_review_length}")
