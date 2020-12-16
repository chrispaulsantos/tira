# importing the libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from datetime import datetime

img_size = 224
dataset_dir = 'dataset'

if __name__ == "__main__":
    print(
        "Num GPUs Available: ", len(
            tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.experimental.list_physical_devices('GPU'))
    

    print('Loading data...')
    # Train data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
    train_ds = train_datagen.flow_from_directory(
        dataset_dir,
        seed=1717,
        target_size=(img_size, img_size),
        batch_size=2, subset='training')

    print(list(train_ds.class_indices.keys()))

    # Test data
    test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    val_ds = test_datagen.flow_from_directory(
        dataset_dir,
        seed=1717,
        target_size=(img_size, img_size),
        batch_size=2, subset='validation')

    classes = list(train_ds.class_indices.keys())

    model = Sequential()
    model.add(layers.Conv2D(input_shape=(img_size,img_size,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096,activation="relu"))
    model.add(layers.Dense(units=4096,activation="relu"))
    model.add(layers.Dense(units=2, activation="softmax"))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    hist = model.fit(
        train_ds,
        steps_per_epoch=100,
        validation_data=val_ds,
        validation_steps=10,
        epochs=11,
        callbacks=[early, tensorboard_callback])

    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
    plt.show()

    test_path = './test.jpg'
    img = keras.preprocessing.image.load_img(
        test_path, target_size=(img_size, img_size)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(classes[np.argmax(score)], 100 * np.max(score))
    )
