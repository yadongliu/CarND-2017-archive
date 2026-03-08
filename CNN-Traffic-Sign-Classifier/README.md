# CNN-Traffic-Sign-Classifier

A Udacity class project using Convolutional Neural Network to classify German traffic signs. 

Training and test data set is from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

The project has the following files: 
* Ipython notebook with code: [Traffic_Sign_Classifier.ipynb](https://github.com/yadongliu/CNN-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)
* Ipython exported as an html file: [Traffic_Sign_Classifier.html](http://htmlpreview.github.io/?https://github.com/yadongliu/CNN-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.html)
* Project write up in markdown: [project_writeup.md](https://github.com/yadongliu/CNN-Traffic-Sign-Classifier/blob/master/project_writeup.md)

## How to Run the Project

The project was created with Python3 & Tensorflow v0.12. You can refer to
[CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for setting up the environment.

```sh
git clone https://github.com/yadongliu/CNN-Traffic-Sign-Classifier.git
cd CNN-Traffic-Sign-Classifier
jupyter notebook Traffic_Sign_Classifier.ipynb
```

You also need to download the traffic sign dataset. The notebook Traffic_Sign_Classifier.ipynb assumes that 
pickled data files (train.p & test.p) reside in a subfolder called "traffic-signs-data". But you are free to change the folder and file names.
