
# coding: utf-8

# **Vehicle Detection Project**
# 
# The goals / steps of this project are the following:
# 
# * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
# * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
# * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# * Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
# * Estimate a bounding box for vehicles detected.
# 

# ### 1 Training a Classifier
# #### 1.1 Data Exploration
# 

# In[308]:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
import pickle
import time
from lesson_functions import *
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

cars = glob.glob('../training_data/vehicles/KITTI_extracted/*.png')
cars += glob.glob('../training_data/vehicles/GTI_Far/*.png')
cars += glob.glob('../training_data/vehicles/GTI_Left/*.png')
cars += glob.glob('../training_data/vehicles/GTI_Right/*.png')
cars += glob.glob('../training_data/vehicles/GTI_MiddleClose/*.png')

notcars = glob.glob('../training_data/non-vehicles/Extras/*.png')
notcars += glob.glob('../training_data/non-vehicles/GTI/*.png')

images = []
for image in images:
    if 'non-vehicles' in image or 'Extra' in image:
        notcars.append(image)
    else:
        cars.append(image)
   
# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    img_sample = mpimg.imread(car_list[0])
    data_dict["image_shape"] = img_sample.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img_sample.dtype
    # Return data_dict
    return data_dict
    
data_info = data_look(cars, notcars)

print('Your function returned a count of', 
      data_info["n_cars"], ' cars and', 
      data_info["n_notcars"], ' non-cars')
print('of size: ',data_info["image_shape"], ' and data type:', 
      data_info["data_type"])
# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])


# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')


# #### 1.2 Feature Extraction
# 
# Image features refer to characteristics that set apart one image from another. One obvious feature is the color value of each pixel or histogram of all color values grouped in ranges (aka bins). A more complicated feature is called HOG, Histogram of Oriented Gradients. We are going to use skimage's hog function to extract the HOG feature from an image. Before calculating the HOG feature, we convert the image from RGB color space to the HLS color space as it handles different lighting situations better. For other HOG parameters, we chose an orientations of 9, pixels_per_cell of 8, and cells_per_block of 2.

# In[326]:


## Testing
car_features = extract_features(cars[:1], hog_channel='ALL')

# #### 1.3 Training 
# 
# For training the classifier, we first scaled the extracted features to zero mean and unit variance with a StandardScaler. We then split our data into 80% training set and 20% test/validation set. Finally we fit a SVM (LinearSVC) model and calculated its accuracy. 

# In[311]:

import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
num_samples = 8000

start = time.time()
car_features = extract_features(cars[0:num_samples], color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars[0:num_samples], color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

print(round(time.time()-start, 2), 'Seconds to extra features from training data ...')
print('car_features: ', len(car_features), ', noncars: ', len(notcar_features))


# In[323]:

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test_val, y_train, y_test_val = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
print('Total training size:', len(X_train))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test_val, y_test_val), 4))

# Check the prediction time for a single sample
t=time.time()
idx_predict = np.random.randint(0, len(X_test_val))
print('My SVC predicts: ', svc.predict(X_test_val[0:10]))
print('For the actual label: ', y_test_val[0:10])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict 10 labels with SVC')

# Saving Classifier Parameters
print('Saving Classifier Parameters to pickle file ...')
pickle_data = {
    'svc': svc,
    'color_space': color_space,
    'orient': orient,
    'pix_per_cell': pix_per_cell,
    'cell_per_block': cell_per_block,
    'hog_channel': hog_channel,
    'spatial_size': spatial_size,
    'hist_bins': hist_bins,
    'spatial_feat': spatial_feat,
    'hist_feat': hist_feat,
    'hog_feat': hog_feat,
    'X_scaler': X_scaler
}
pickle_file = 'VehicleClassifier.p'
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(pickle_data, pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
print('Data cached in pickle file.')