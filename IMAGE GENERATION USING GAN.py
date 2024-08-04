import tensorflow as tf
from tensorflow import keras
import numpy as np

(X_train, _), (_, _) = keras.datasets.mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5 
X_train = np.expand_dims(X_train, axis=-1)

generator = keras.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=(100,)),
    keras.layers.Reshape((7, 7, 128)),
    keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='tanh')
])

discriminator = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(28, 28, 1)),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
    keras.layers.LeakyReLU(alpha=0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation='sigmoid')
])
discriminator.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                      metrics=['accuracy'])
discriminator.trainable = False

gan_input = keras.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.0002))
batch_size = 64
epochs = 10000
sample_interval = 1000
for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise)
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, real_labels)
    if epoch % sample_interval == 0:
        print(f'Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}')
    _, accuracy = discriminator.evaluate(np.concatenate([real_images, fake_images]),
                                         np.concatenate([real_labels, fake_labels]), verbose=0)
    print(f"Discriminator Accuracy: {accuracy:.4f}")

