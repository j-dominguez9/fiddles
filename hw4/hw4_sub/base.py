import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

now = datetime.datetime.now
batch_size=128
num_classes = 10
epochs = 25


img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


def train_model(model, train, test, num_classes):
    X_train = train[0].reshape((train[0].shape[0],) + input_shape)
    X_test = test[0].reshape((test[0].shape[0],) + input_shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train.shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
    t = now()
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data=(X_test, y_test))
    print('Training time: %s' % (now() - t))
    model.save('mnist_model.tf')
    score = model.evaluate(X_test, y_test, verbose = 0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# create two datasets one with digits below 5 and one with 5 and above
#X_train_lt5 = X_train[y_train < 5]
#y_train_lt5 = y_train[y_train < 5]
#X_test_lt5 = X_test[y_test < 5]
#y_test_lt5 = y_test[y_test < 5]

#X_train_gte5 = X_train[y_train >= 5]
#y_train_gte5 = y_train[y_train >= 5] - 5
#X_test_gte5 = X_test[y_test >= 5]
#y_test_gte5 = y_test[y_test >= 5] - 5


# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

# create complete model
model = Sequential(feature_layers + classification_layers)


train_model(model, (X_train, y_train), (X_test, y_test), num_classes)
#print(X_train.shape)
#print(y_train.shape)

## train model for 5-digit classification [0..4]
#train_model(model,
#            (X_train_lt5, y_train_lt5),
#            (X_test_lt5, y_test_lt5), num_classes)

## freeze feature layers and rebuild model
#for l in feature_layers:
#    l.trainable = False

## transfer: train dense layers for new classification task [5..9]
#train_model(model,
#            (X_train_gte5, y_train_gte5),
#            (X_test_gte5, y_test_gte5), num_classes)

#def main():
#    print(len(X_test))

#if __name__ == "__main__":
#    main()

