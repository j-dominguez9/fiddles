import numpy as np
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras import backend as K

batch_size = 100


def load():
    """
    Loads previous model used as base for transfer learning and new image dataset; returns base model, train/test, and class_names
    """
    # import trained model
    model = tf.keras.models.load_model("mnist_model.tf")
    # import dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "resized_images",
        labels="inferred",
        validation_split=0.2,
        subset="training",
        seed=9,
        image_size=(28, 28),
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "resized_images",
        labels="inferred",
        validation_split=0.2,
        subset="validation",
        seed=9,
        image_size=(28, 28),
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical",
    )
    class_names = np.array(train_ds.class_names)
    return model, train_ds, val_ds, class_names
    # print(class_names)


def input():
    """
    returns input_shape variable used in later function
    """
    img_rows, img_cols = 28, 28

    if K.image_data_format() == "channels_first":
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    return input_shape, img_rows, img_cols


def preprocess(train, test, img_rows, img_cols):
    """
    takes in train and test and normalizes data
    """
    # normalize and augment
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    # data_augmentation = Sequential([tf.keras.layers.RandomFlip('horizontal', input_shape = (img_rows, img_cols, 1)),tf.keras.layers.RandomRotation(0.1),])
    # apply layers
    train_data = train.map(lambda x, y: (normalization_layer(x), y))
    test_data = test.map(lambda x, y: (normalization_layer(x), y))
    # let_data = data.map(lambda x, y: (data_augmentation(x), y))
    return train_data, test_data


# result_batch = model.predict(let_data)
# print(tf.math.argmax(result_batch, axis=-1))


def set_trainability(model):
    """
    makes base model headless for transfer learning
    """
    for i in range(10):
        model.layers[i].trainable = False
    for i in range(10, 12):
        model.layers[i].trainable = True
    return model


def train_model(epochs, model, names, train, test):
    """
    creates layers, adds base model + new layers, compiles new model, and fits new model; returns new model, prints summary of base and new model
    """
    num_classes = len(names)
    class_layer = [
        Dense(300, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]

    model_2 = Sequential([model] + class_layer)
    model.summary()
    model_2.summary()
    # preds = model_2(image_batch)
    # print(preds.shape)
    model_2.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model_2.fit(train, validation_data=test, epochs=epochs, verbose=1)
    return model_2


def predictions(fit_model, class_names, image_batch):
    """
    prints class names of predictions
    """
    predicted_batch = fit_model.predict(image_batch)
    predicted_id = tf.math.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]
    print(predicted_label_batch)


def main():
    model, train_ds, val_ds, class_names = load()
    input_shape, img_rows, img_cols = input()
    train_ds, val_ds = preprocess(train_ds, val_ds, img_rows, img_cols)
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        # print(image_batch)
        print(f"unprocessed: {labels_batch.shape}")
        # print(labels_batch)
        break
    model = set_trainability(model)
    model_2 = train_model(80, model, class_names, train_ds, val_ds)
    predictions(model_2, class_names, image_batch)


if __name__ == "__main__":
    main()
