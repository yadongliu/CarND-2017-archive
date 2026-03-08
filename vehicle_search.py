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

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles
from lesson_functions import *

import pickle
from scipy.ndimage.measurements import label

import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# ### 2 Sliding Window Search on Single Image
# 
# #### 2.1 Sliding Window Feature Extraction and Search
# Now we trained a classifier, we can use it to search for cars on a single image. First, we need a function to create a list of search windows (i.e. bounding boxes). Then we need a function to extract features from each window. We pass the extracted features to our classifier to make a prediction. 
# 
# A few considerations here are important to make searching work. First is the fact we don't need the sliding windows to cover the entire image as we are not interested in cars in the sky (not yet). We can safely ignore the upper half of the image. Similarly we can ignore a part of the bottom as well because that was mostly the dashboard. This reduces the number of windows to search and speed up the algorithm. The second consideration is how much overlapping is necessary between windows. If the overlap is too small, you may miss areas of interests. If the overlap is too large, you end up with a lot of windows to search. Unfortunately it requires a lot of trial and error to choose the right set of parameters such as y start/stop position and overlap ratio. 
# 
# Last but not the least, we used four different window sizes (192x192, 128x128, 96x96, 64x64) to perform the sliding search. We created a function called `multi_scale_search()` to make this process easier. Through trial and error, we found that an 75% overlap worked the most reliably. 
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan)/nx_pix_per_step) 
    ny_windows = np.int((yspan)/ny_pix_per_step) 
    #nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    #ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 

    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
def multi_scale_search(image, x_start_stops, y_start_stops, xy_windows, xy_overlaps, box_colors, debug_boxes=False):
    hot_windows = []
    all_windows = []
    if debug_boxes == True:
        draw_image = np.copy(image)

    for i in range(len(x_start_stops)):
        windows = slide_window(image, x_start_stop=x_start_stops[i], y_start_stop=y_start_stops[i], 
                    xy_window=xy_windows[i], xy_overlap=xy_overlaps[i])
        all_windows += [windows]
        hot_boxes = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
        hot_windows += hot_boxes
        if debug_boxes == True:
            draw_image = draw_boxes(draw_image, hot_boxes, color=box_colors[i], thick=3) 

    if debug_boxes == True:      
        return hot_windows, all_windows, draw_image
    else:
        return hot_windows, all_windows

## Helper
def draw_two_imgs(img1, img2, title1='Original Image', title2='Undistorted Image'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=24)
    ax2.imshow(img2)
    ax2.set_title(title2, fontsize=24)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# Read back classifier data
with open('VehicleClassifier.p', mode='rb') as f:
    data = pickle.load(f)

svc = data['svc']
color_space = data['color_space']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data['cell_per_block']
hog_channel = data['hog_channel']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
spatial_feat = data['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']
X_scaler = data['X_scaler']

image = mpimg.imread('test_images/test1.jpg')
# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

x_start_stops = [[None, None],[None, None],[None, None], [None, None]]
y_start_stops = [[380, 480], [390, 462], [400, 448], [410, 442]] # Min and max in y to search in slide_window()
xy_windows = [[200, 200], [144, 144], [96, 96], [64, 64]]
xy_overlaps = [(0.75, 0.75), (0.75, 0.75), (0.75, 0.75), (0.75, 0.75)]
box_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

t = time.time()
hot_windows, all_windows, image_carbox = multi_scale_search(image, x_start_stops, y_start_stops, 
                   xy_windows, xy_overlaps, box_colors, debug_boxes=True)
win_count = len(all_windows[0])+len(all_windows[1])+len(all_windows[2])+ len(all_windows[3])
print(round(time.time()-t, 2), 'Seconds to search ', win_count ,' windows: ')

candidate_windows = []
for w in all_windows:
    candidate_windows += w

'''
t = time.time()
allwindow_draw = np.copy(image)
for i in range(len(all_windows)):
    allwindow_draw = draw_boxes(allwindow_draw, all_windows[i], color=box_colors[i], thick=6)                    
draw_two_imgs(image_carbox, allwindow_draw, title1='Car Box', title2='All Windows')
print(round(time.time()-t, 2), 'Seconds to draw_boxes ...')
'''

## Run sliding window search on more test images
def test_more_images():
    test_imgs = glob.glob('test_images/*.jpg')
    output_dir = 'output_images/out_'
    for file in test_imgs:
        image = mpimg.imread(file)
        hot_windows = search_windows(image, candidate_windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, hog_channel=hog_channel, 
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        # output_file = output_dir+file.split('/')[-1]
        # mpimg.imsave(output_file, image_carbox)


## test_more_images()

'''
out_files = glob.glob('output_images/*.jpg')
fig, axarr = plt.subplots(2, 3, figsize=(18, 9))

for index, file in enumerate(out_files):
    fig.tight_layout()
    r = np.int(index / 3)
    c = index % 3
    image = mpimg.imread(file)
    axarr[r, c].imshow(image)
    title = "Image {0}".format(index+1)
    axarr[r, c].set_title(title)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''

# #### 2.2 Remove False Positives
# 
# We will use the heatmap technique to remove false positive from our prediction.

# Read in a pickle file with bboxes saved
# Each item in the "all_bboxes" list will contain a 
# list of boxes for one of the images shown above
#box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img

heat = np.zeros_like(image[:,:,0]).astype(np.float)
heat = add_heat(heat, hot_windows)
heat = apply_threshold(heat, 0)
heatmap = np.clip(heat, 0, 255)
# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

'''
fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
'''


# ### 3 Sliding Window Search on Video
# 
# Now put everything together, we create our image process pipeline via the `process_image` function. We are going to use a helper class to keep track of the hot windows from the last `n` video frames, in order to have a more reliable algorithm to remove false positives through heatmap/thresholding. 
# 
# The result of running the process pipeline on project_video.mp4 is [project_video_out.mp4](project_video_out.mp4)

from collections import deque

class DetectionBoxes:
    def __init__(self, n):
        self.n = n
        self.current_frame = None
        self.queued_frames = deque([], maxlen=n)
        self.all_boxes = []

    def add_frame(self, boxes):
        self.current_frame = boxes
        self.queued_frames.appendleft(boxes)
        #self.get_all_boxes()
        all_boxes = []
        for boxes in self.queued_frames:
            all_boxes += boxes
        if len(all_boxes) > 0:
            self.all_boxes = all_boxes
        else:
            self.all_boxes = []

#print('Hot windows detected: ', len(hot_windows),', out of ', len(candidate_windows))
detector = DetectionBoxes(1)
def process_image(image):
    # take a image and search for cars
    ## draw_image = np.copy(image)
    ## image = image.astype(np.float32)/255
    hot_windows = search_windows(image, candidate_windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, hog_channel=hog_channel, 
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    # detector.add_frame(hot_windows)
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    # heatmap = add_heat(heatmap, detector.all_boxes)
    heatmap = add_heat(heatmap, hot_windows)
    heatmap  = apply_threshold(heatmap, 0)
    labels = label(heatmap)
    draw_image = draw_labeled_bboxes(image, labels)
    return draw_image


'''
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#video_output = 'project_video_out.mp4'
#clip1 = VideoFileClip("project_video.mp4")
video_output = 'test_video_out.mp4'
clip1 = VideoFileClip("test_video.mp4")
processed_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
processed_clip.write_videofile(video_output, audio=False)
'''

# get_ipython().magic('time processed_clip.write_videofile(video_output, audio=False)')


# ### 4 Discussions
# 
# This is another fun project to work on where we got a chance to experiment with advanced CV algorithm (HOG extraction) and the simplicity and power of using scikit-learn's SVM module. Once you gathered all of your training images (vehicles v.s. non-vehicles), using extracted features to train the SVM is a fairly straightforward process.   
# 
# One of the more tedious aspect of this project is to fine-tune the various parameters in sliding window search. The window size, start & stop positions of the search area, overlapping ratios of the sliding windows can all impact the performance and reliability of the algorithm. Unfortunately there is no shortcut to optimization. You would just have to take the time to trial and error. 
# 
# Because we used multiple scales in sliding window search, one potential area for future work is how to determine if multiple neighboring boxes are actually identifying the same object (cars in our case) and combine these boxes into one.  

# In[ ]:



