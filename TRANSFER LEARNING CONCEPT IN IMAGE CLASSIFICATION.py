import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt

class_names = ['Cats', 'Dogs']  # Update with your actual class names

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
    validation_split=0.2  # Adding validation split
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # ResNet50 input size
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # ResNet50 input size
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(len(class_names), activation='softmax')(x)

transfer_model = models.Model(inputs=base_model.input, outputs=predictions)
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
transfer_model.summary()

print("Training started...")
history = transfer_model.fit(train_generator, epochs=10, validation_data=validation_generator)
print("Training completed.")

print("Saving the model...")
transfer_model.save(r'C:\Users\LENOVO\PycharmProjects\nn\transfer_learning_resnet50_model.h5')
print("Model saved successfully.")

print("Making predictions...")
img_path = r'C:\Users\LENOVO\PycharmProjects\nn\pet.jpg'  # Update with the path to the image you want to classify
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # Resize images to match the input size expected by ResNet50
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values to [0, 1]

predictions = transfer_model.predict(img_array)
predicted_class = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class]

plt.imshow(img)
plt.axis('off')
plt.title('Predicted Class: {}'.format(predicted_class_name))
plt.show()
print("Prediction completed.")
