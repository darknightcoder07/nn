import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
(X_train_all, y_train_all), (X_test, y_test) = mnist.load_data()
X_valid = X_train_all[:10000]
y_valid = y_train_all[:10000]
X_train = X_train_all[10000:]
y_train = y_train_all[10000:]
X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype("float32") / 255
X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype("float32") / 255
X_valid = X_valid.reshape((X_valid.shape[0], 28 * 28)).astype("float32") / 255
plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.axis('off')
    plt.imshow(X_train_all[10000 + i], cmap='gray')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()
