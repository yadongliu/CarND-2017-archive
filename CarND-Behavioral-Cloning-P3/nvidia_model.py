import numpy as np
from six.moves import cPickle as pickle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Activation, Dropout
from keras.layers import Cropping2D, Lambda

crop_values=[0, 0, 50, 20]
img_width = 160
img_height = 320
# Initialize the model and crop image
model = Sequential()
model.add(Cropping2D(cropping=((crop_values[0], crop_values[1]), (crop_values[2], crop_values[3])) ,\
                             input_shape=(img_width, img_height, 3)))

model.add(Lambda(lambda x: x/255 - .5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))

model.add(Flatten())
# add in dropout of .5 (not mentioned in Nvidia paper)
# model.add(Dropout(.5))
# model.add(Activation('relu'))

model.add(Dense(1164, activation='relu'))
model.add(Dense(100))
# model.add(Dropout(.3))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# load data from pickle files for later access
with open('steering.p', 'rb') as f:
    y_train = pickle.load(f)

with open('train.p', 'rb') as f:
    X_train = pickle.load(f)

print(X_train.shape)
print(y_train.shape)

model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save('nvidia_model.h5')
