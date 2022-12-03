import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D
from keras.layers import MaxPool2D,Flatten,Dropout,ZeroPadding2D,BatchNormalization
from keras.utils import np_utils

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keepdims=False)
    return outputs

def createMaxout():
        # Build model
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1200)(x)
    x = max_out(x, 300, axis=None)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(40)(x)
    x = max_out(x, 10, axis=None)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model

def createCNN():
    pool_size = (2,2)

    CNN = Sequential()

    CNN.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),input_shape=(28,28,1)))
    CNN.add(Activation('relu'))
    CNN.add(MaxPool2D(pool_size=pool_size))

    CNN.add(ZeroPadding2D((1,1)))
    CNN.add(Conv2D(48,kernel_size=(3,3)))
    CNN.add(Activation('relu'))
    CNN.add(BatchNormalization(epsilon=1e-6,axis=1))
    CNN.add(MaxPool2D(pool_size=pool_size))

    CNN.add(ZeroPadding2D((1,1)))
    CNN.add(Conv2D(64,kernel_size=(2,2)))
    CNN.add(Activation('relu'))
    CNN.add(BatchNormalization(epsilon=1e-6,axis=1))
    CNN.add(MaxPool2D(pool_size = pool_size))

    CNN.add(Dropout(0.25))
    CNN.add(Flatten())

    CNN.add(Dense(3168))
    CNN.add(Activation('relu'))

    CNN.add(Dense(10))
    CNN.add(Activation('softmax'))
    CNN.summary()
    return CNN