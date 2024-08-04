import tensorflow as tf
import numpy as np
num_users = 1000
num_items = 500
num_samples = 10000
user_ids_train = np.random.randint(0, num_users, num_samples)
item_ids_train = np.random.randint(0, num_items, num_samples)
ratings_train = np.random.randint(1, 6, num_samples) 
user_ids_val = np.random.randint(0, num_users, num_samples)
item_ids_val = np.random.randint(0, num_items, num_samples)
ratings_val = np.random.randint(1, 6, num_samples)
user_ids_test = np.random.randint(0, num_users, num_samples)
item_ids_test = np.random.randint(0, num_items, num_samples)
ratings_test = np.random.randint(1, 6, num_samples)
class CollaborativeFilteringModel(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
        self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
        self.dot = tf.keras.layers.Dot(axes=1)
    def call(self, inputs):
        user_id, item_id = inputs
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        return self.dot([user_embedding, item_embedding])
embedding_size = 50
model = CollaborativeFilteringModel(num_users, num_items, embedding_size)
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit([user_ids_train, item_ids_train], ratings_train,
                    validation_data=([user_ids_val, item_ids_val], ratings_val),
                    epochs=10, batch_size=64)
loss = model.evaluate([user_ids_test, item_ids_test], ratings_test)
print("Test Loss:", loss)
