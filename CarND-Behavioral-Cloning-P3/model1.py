import numpy as np
from six.moves import cPickle as pickle

#X_train = np.array(images)
#y_train = np.array(measurements)

# load data from pickle files for later access
with open('steering.p', 'rb') as f:
    y_train = pickle.load(f)

with open('train.p', 'rb') as f:
    X_train = pickle.load(f)

print(X_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
