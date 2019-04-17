# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


def resize(img):
    """
    Resizes the images in the supplied tensor to the original dimensions of the NVIDIA model (66x200)
    """
    from keras.backend import tf as ktf
    return ktf.image.resize_images(img, [66, 200])


def network_model(dropout=0.25):
    model = Sequential()
    # Cropping image
    model.add(Lambda(lambda imgs: imgs[:, -150:, :, :], input_shape=(240, 320, 3)))
    # Normalise the image - center the mean at 0
    model.add(Lambda(lambda imgs: (imgs / 255.0) - 0.5))
    model.add(Lambda(resize))

    # We have a series of 3 5x5 convolutional layers with a stride of 2x2
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(BatchNormalization())

    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(BatchNormalization())

    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(BatchNormalization())

    # This is then followed by 2 3x3 convolutional layers with a 1x1 stride
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    model.add(BatchNormalization())

    # Flattening the output of last convolutional layer before entering fully connected phase
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1))

    model.compile(loss="MSE", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model