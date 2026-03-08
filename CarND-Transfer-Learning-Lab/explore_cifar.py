from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.datasets import cifar10
from keras.optimizers import Adam
from sklearn.utils import shuffle
import numpy as np

def check_layers(layers, true_layers):
    assert len(true_layers) != 0, 'No layers found'
    for layer_i in range(len(layers)):
        assert isinstance(true_layers[layer_i], layers[layer_i]), 'Layer {} is not a {} layer'.format(layer_i+1, layers[layer_i].__name__)
    assert len(true_layers) == len(layers), '{} layers found, should be {} layers'.format(len(true_layers), len(layers))

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

X_train, y_train = shuffle(X_train, y_train)
nb_classes = len(np.unique(y_train))

print('X_train[0]:', X_train[0].shape)
print('y_train:', y_train.shape)
print('y_test: ', y_test.shape)

# Construct the network and add a pooling layer after the convolutional layer.

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32, 32, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Flatten(input_shape=(16, 16, 3)))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

check_layers([Convolution2D, MaxPooling2D, Activation, Flatten, Dense, Activation, Dense, Activation], model.layers)
assert model.layers[1].pool_size == (2, 2), 'Second layer must be a max pool layer with pool size of 2x2'

# TODO: Compile and train the model here.
# Configures the learning process and metrics
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=4, validation_split=0.2)
# assert(history.history['val_acc'][-1] > 0.91), "The validation accuracy is: %.3f.  It should be greater than 0.91" % history.history['val_acc'][-1]
# print('Tests passed.')

# Calculate test score
# test_score = model.evaluate(X_test_normalized, y_test_one_hot[0:12630])