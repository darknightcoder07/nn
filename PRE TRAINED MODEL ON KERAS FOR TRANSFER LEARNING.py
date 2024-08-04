import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Cats', 'Dogs']

train_dir = r'C:\Users\LENOVO\PycharmProjects\nn\train'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Added validation split
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Use 'binary' for binary classification
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Use 'binary' for binary classification
    subset='validation'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation='sigmoid')(x)  # Binary classification, so 1 output neuron with sigmoid activation

transfer_model = models.Model(inputs=base_model.input, outputs=predictions)
transfer_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

transfer_model.summary()

print("Training started...")
history = transfer_model.fit(train_generator, epochs=10, validation_data=validation_generator)
print("Training completed.")

print("Saving the model...")
transfer_model.save(r'C:\Users\LENOVO\PycharmProjects\nn\transfer_learning_model1.keras')
print("Model saved successfully.")

img_path = r'C:\Users\LENOVO\PycharmProjects\nn\pet.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values to [0, 1]

print("Making predictions...")
predictions = transfer_model.predict(img_array)
predicted_class = int(predictions[0][0] > 0.5)  # Binary classification threshold
predicted_class_name = class_names[predicted_class]

plt.imshow(img)
plt.axis('off')
plt.title('Predicted Class: {}'.format(predicted_class_name))
plt.show()
