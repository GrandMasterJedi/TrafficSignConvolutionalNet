[//]: # (Image References)

[image1]: ./examples_output/2_Rand10images.png "Rand10"
[image2]: ./examples_output/3_SampleDistribution.png "Histogram"
[image3]: ./examples_output/4_ValidationAccuracy.png "Valid"
[image4]: ./examples_output/5_SignsFromWeb.png "ExtraSigns"






# **Traffic Sign Recognition**  

I build a learner to classify traffic signs from the German dataset using Convolutional Neural Networks as implemented in Tensor Flow. The learner is based on the LeNet5 architecture of LeCun, Y. (2013) 

The validation accuracy is 98%, the testing accuracy is 94%. On a new dataset of 10 traffic signs I downloaded from the web the prediction accuracy is 60%

This project is my solution to assignment (1.2) of the Udacity Self Driving Car Nanodegree. 

---
## Dependencies
* Python 3.x
* NumPy
* OpenCV
* Matplotlib
* Pandas
* math
* time

---
## Goals / Steps
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. 

### Writeup / README
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You are reading it and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Below is 10 randomly selected signs from the German traffic signs dataset. As the size of each image is (32, 32), they have relatively low resolution.
![alt text][image1]

As an exploratory visualization of sample distribution among the classes, I plot the distribution histogram below. The sample of sign is not balanced, as some signs are more representative then others. However this representativeness proportion is kept among training, validation and testing sample. 

As the sample is not uniformly distributed among the classes it is possible that for some classes we can achieve better prediction than for other classes. As proportions of signs among training, validation and testing samples are relatively constant, we can expect the performance of between training, validation and between training and testing not being biased by sample selection
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For data preprocessing I allow the option to convert the images to grayscale. This reduces the input data by averaging the 3 color channels (RGB) into one channel. However the validation performance is not necessarily better, as colors in traffic sign (especially red) have a great significance. In the final model I choose, I leave the three channels data as input. The images are normalized by the conventional way of detracting and dividing each pixel x by 128: NormalizedPix = (Pix-128)/128. This can smooth the effect of outlier pixel values.

Overall, not much data preprocessing is applied.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0. Input         		| 32x32x3 RGB image   							| 
| 1. Convolution 5x5    | 1x1 stride, valid padding, 	outputs 28x28x9	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 					outputs 14x14x9	|
| 2. Convolution 3x3	| 1x1 stride, valid padding, 	outputs 12x12x27|
| RELU					|												|
| 3. Convolution 4x4    | 1x1 stride, valid padding, 	outputs 9x9x81 	|
| RELU					|												|
| Max pooling	      	| 3x3 stride   					outputs 3x3x81  |
| Flatten				|        						output  729x1	|
| 4. Fully Connected	|         			 			outputs 291x1 	|
| RELU					|												|
| 5. Fully Connected	|        						outputs 116x1 	|
| RELU					|												|
| 6. Fully Connected	| outputs number of classes 			43x1	|
| Softmax				|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The following are the training parameters I use:
``
EPOCHS = 40
BATCH_SIZE = 256 
nchannel = 3
Optimizer = tf.train.AdamOptimizer(learning_rate = 0.002)
``

In chosing those parameter values, I reference the experience in training the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) with the LeNet5 architecture. In that case 10 EPOCHS and batch size of 128
are used and with the Adam (Kingma, Ba, 2014) algorithm the validation performance was 99%. The convolutional network architecture used here is similar to the LeNet5, except that it adds one additional convolutional layer. For handwritten digits, there were only 10 classes and images had one color channel. In this case, for the traffic signs, we have 43 classes with 3 color channels. Therefore I chose higher sampling complexity.

The Adam algorithm stands for adaptive moment. It is a stochastic gradient algorithm that weighs the gradient with its first order moment at each step. I set the learning rate to 0.002, that is somewhat high, to speed up computational time.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I tried first the LeNet5 with grayscale signs, achieving only around 80% validation accuracy. Then I chose RGB input channels and increase the filter depth to be multiple 3. This increase the accuracy. I add a convolutional layer, decrease all layer patches to (5x5), (3x3), (4x4) and try both average pooling and max pooling, leading to the current model with validation accuracy 98%.

The validation accuracy for the different EPOCHS is plotted below.
![alt text][image3]

Overall, I end up with the current architecture based on intuition (that may be well wrong) rather than solid scientific basis.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I download 10 images from the web (instead of 5), cropped them and saved them as .png in the [example folder](examples/). The images are found by typing German Traffic Signs in Google image search. They are randomly selected and have a relatively high quality. I then resize them to (32x32). Below are the 10 signs.
![alt text][image4]

Two images belong to the same class and three are speed limit signs. Such a small sample is very unbalanced and can lead to very good or very bad prediction performance.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	  				| 
|:---------------------:|:---------------------------------:| 
| Speed Limit 50      	| Children Crossing   				| 
| Pedestrian Crossing   | General Caution 					|
| No Entry				| No Entry							|
| Pedestrian Crossing	| General Caution	 				|
| Speed Limit 120		| Roundabout Mandatory				|
| Stop      			| Stop  							| 
| Road Work    			| Road Work 						|
| Roundabout Mandatory	| Roundabout Mandatory				|
| Speed Limit 70	    | Speed Limit 70					|
| Yield					| Yield      						|


The model was able to correctly guess 6 of the 10 traffic signs, which gives an accuracy of 60%. This does not compares favorably to the accuracy on the test set of 94%. Two _speed limit_ signs out of three are miss-predicted and the two _pedestrian crossing_ signs are all miss-predicted. 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the last cell of the Ipython notebook. The softmax probability for all traffic sign predictions are very high for the top 1 prediction. Below is prediction table with probability for the top 5 prediction:

| Image			        |  Top 1 Prediction	  		|	Prob. of Top 5 Predictions 				| 
|:---------------------:|:-------------------------:|:-----------------------------------------:|
| Speed Limit 50      	| Children Crossing   		| 99.99% 	0%		0% 		0% 		0%		|
| Pedestrian Crossing   | General Caution 			| 99.97% 	0.03% 	0% 		0% 		0%		|
| No Entry				| No Entry					| 100%		0%		0% 		0% 		0%		|
| Pedestrian Crossing	| General Caution	 		| 87.56% 	6.74% 	4.73% 	0.86% 	0.09%	|
| Speed Limit 120		| Roundabout Mandatory		| 73.88%	21.81%	4.2%	0.06%   0.03%	|
| Stop      			| Stop  					| 100%		0%		0% 		0% 		0% 		|
| Road Work    			| Road Work 				| 100%		0%		0% 		0% 		0%		|
| Roundabout Mandatory	| Roundabout Mandatory		| 100%		0%		0% 		0% 		0%		|
| Speed Limit 70	    | Speed Limit 70			| 100%		0%		0% 		0% 		0%		|
| Yield					| Yield      				| 100%		0%		0% 		0% 		0%		|

When it is correctly predicted the prediction have 100% probability. However even when the prediction have 99.99% probability, the ground true is not correctly predicted as in the first image of speed limit. This pattern is interesting. I leave it open to suggestion for why it is the case.




---
## Resources
* Udacity Self-Driving Car [Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) 
* Udacity project assignment and template on [GitHub](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)
* Udacity project [rubric](https://review.udacity.com/#!/rubrics/481/view)

