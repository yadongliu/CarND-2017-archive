import csv
import cv2
import numpy as np
from six.moves import cPickle as pickle

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    last_component = source_path.split('/')[-1]
    local_path = '../data/IMG/' + last_component
    image = cv2.imread(local_path)
    images.append(image)
    # steering angle measurements
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)
print(y_train.shape)

# Save to pickle file for later access
with open('steering.p', 'wb') as f:
    pickle.dump(y_train, f, pickle.HIGHEST_PROTOCOL)

with open('train.p', 'wb') as f:
    pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)

