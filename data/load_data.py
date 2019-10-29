from __future__ import print_function
import math
import os
from io import open
import librosa
import struct
import numpy
import glob
import array
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import os
import sys
from tensorflow.python.client import device_lib

def load_data(file):
    with open(file, 'rb') as data_file:
        bands = struct.unpack('i', data_file.read(struct.calcsize('i')))[0]

        while True:
            read = data_file.read(struct.calcsize('f'))

            if not read:
                break

            y = struct.unpack('f', read)[0]
            length = struct.unpack('i', data_file.read(struct.calcsize('i')))[0]
            spec = []

            for i in range(bands):
                line = array.array('f')
                line.fromstring(data_file.read(struct.calcsize('f'*length)))

                spec.append(line)

            x = numpy.array(spec)

            yield [x, y]


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    sys.exit()

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for data in load_data('./data_train.dat'):
        x_train.append(data[0])
        y_train.append(data[1])

    x_train = numpy.asarray(x_train)
    y_train = numpy.asarray(y_train)

    for data in load_data('./data_test.dat'):
        x_test.append(data[0])
        y_test.append(data[1])

    x_test = numpy.asarray(x_test)
    y_test = numpy.asarray(y_test)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    batch_size = 1
    epochs = 1

    img_x, img_y = 1293, 128

    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

    input_shape = (img_x, img_y, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu'))

    model.add(Flatten())
    model.add(Dense(128*1293, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

