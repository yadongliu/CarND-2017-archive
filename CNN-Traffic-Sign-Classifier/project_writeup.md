#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/dataset_distribution.jpg "Distribution"
[image2]: ./images/random_samples.jpg "Random Samples"
[image3]: ./images/test_images.jpg "Test Images"
[image4]: ./images/prediction_probability.jpg "Prediction Probability"

## Rubric Points

---
###Project Files

File List | Links
------------ | -------------
Ipython notebook with code | [Traffic_Sign_Classifier.ipynb](https://github.com/yadongliu/CNN-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)
Ipython exported as an html file | [Traffic_Sign_Classifier.html](http://htmlpreview.github.io/?https://github.com/yadongliu/CNN-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.html)
Project write up in markdown | [project_writeup.md](https://github.com/yadongliu/CNN-Traffic-Sign-Classifier/blob/master/project_writeup.md)

---
###Data Set Summary & Exploration

What | Explanation
------------ | -------------
Dataset Summary | The dataset are loaded from pickle files, one file for each type of data: training.p, valid.p and test.p <br> * Number of training examples = 34799 <br> * Number of validation examples = 4410 <br> * Number of testing examples = 12630<br> * The shape of a traffic sign image is 32x32x3 <br> * The number of unique classes/labels in the data set is 43
Dataset Exploration | Some exploration steps include: <br> * Look at how number of samples are distributed across the 43 classes via a scatter plot <br> * Shows a random sample of 9 images from the training dataset along with their labels/classes

Here is an exploratory visualization of the data set. Here is a scatter plot showing how the data are distributed across different labels. 

![Distribution][image1]

Here is a random sample of 9 images from the training dataset:

![alt text][image2]

###Design and Test a Model Architecture

####1. The pre-processing on the dataset I tried include normalization and data augmentation.

Normalization refers to the operation of converting pixel data (0 - 255) to the range (-0.5, 05). This seems to help the training process to converge faster.

I also tried data augmentation by using Tensorflow's Images API, including tf.images.random_flip_up_down and tf.images.random_brightness function during batch training. However, this slows down the training considerably, like 10x slower. In the end, I didn't include data augmentation. For future work, data augmentation could be done separately from training and save the augmented data.  

####2. The LeNet lab model was used as a starting point for this project. 

A few minor changes were made the LeNet model. One is to increase the number of neurons in the fully connected hidden layers. The first and second has 240 and 86 nodes respectively. The second change made was to include dropout to reduce overfitting. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, 'VALID' padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten/Fully connected	|  input 400       									|
| First Hidden Layer |  120    |
| RELU					|												|
| Dropout					|					keep_prob: 0.5							|
| Second Hidden Layer |  86    |
| RELU					|												|
| Dropout					|					keep_prob: 0.5							|
| Output	Layer			|     43   									|
|	Loss function					|			softmax_cross_entropy_with_logits									|
|	Optimizer					|		AdamOptimizer										|
 


####3. Optimizer and Hyperparameters

AdamOptimizer is used to minimize a loss function softmax_cross_entropy_with_logits. 

Batch size = 128
Learning rate = 0.001
EPOCHS = 40

Beyond 40 epochs, the training accuracy and validaton accuracy don't improve much more. 

My final model results were:
* training set accuracy of: 0.984
* validation set accuracy of: 0.933
* test set accuracy of: 0.916

The result seems to indicate that there might still be slightly overfitting issue with the model. Using data augmentation techniques could probably help with this. 

###Test a Model on New Images

####1. Here are six German traffic signs found on the web and will be used to test the classifier model.

![alt text][image3]

####2. Discussing Prediction Result

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road (label: 22)    		| Bumpy road  									| 
| Children crossing (label: 28)	| Children crossing										|
| No entry			(label: 17)			| No entry											|
| Road work  (label: 25)		|  Road work     |
| 30 km/h	   (label: 1)	 		| 30 km/h				 				|
| 50 km/h	   (label: 2)	 		| 50 km/h					 				|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91.6%. 

####3. The model seems very certain when predicting on each of the fix new images as can be seem by plotting the softmax probabilities for each prediction shown below. 

![alt text][image4] 

Here is for the first image. We can see that the prediction model is 99.99% certain the image is a "Bumpy road" sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999         			| Bumpy road    									| 
| 2.619e-07     				| Traffic signals 										|
| 7.9256e-12					| Keep right											|
| 6.0400e-12	      			| Bicycles crossing					 				|
| 3.1144e-13				    | Wild animals crossing      							|

For the rest of test images, the model is similarly very certain (over 99%) in its predictions. 

In the future, it might be worthwhile to intentionally find traffic signs that are under-presented in the training data and test the model's prediction. 
