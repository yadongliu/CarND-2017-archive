import os
import csv

samples = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)  # skip the headers
	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import random
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Activation, Dropout
from keras.layers import Cropping2D, Lambda

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # left camera
                name = '../data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = center_angle + 0.25
                images.append(left_image)
                angles.append(left_angle)
                # right camera
                name = '../data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = center_angle - 0.25
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print('y_train', y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format
crop_values=[50, 20, 0, 0]

model = Sequential()
model.add(Cropping2D(cropping=((crop_values[0], crop_values[1]), (crop_values[2], crop_values[3])) ,\
                             input_shape=(row, col, ch)))
# Preprocess incoming data, centered around zero with small standard deviation 
# model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
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
model.summary()

model.compile(loss='mse', optimizer='adam')
nb_spe =len(train_samples)*3
nb_val = len(validation_samples)*3
history = model.fit_generator(train_generator, samples_per_epoch=nb_spe, validation_data=validation_generator, nb_val_samples=nb_val, nb_epoch=12)
print(history.history.keys())
model.save('model_gen.h5')
print(history.history['loss'])
print(history.history['val_loss'])

def show_history(history):
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()