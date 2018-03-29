[//]: # (Image References)

[image1]: ./examples_output/2_Rand10images.png "Rand10"
[image2]: ./examples_output/3_SampleDistribution.png "Histogram"






# **Traffic Sign Recognition**  

I build a lerner to classify traffic signs from the German dataset using Convolutional Neural Networks as implemented in Tensor Flow. The learner is based on the LeNet5 architecture of LeCun, Y. (2013) 

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

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)


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

As an exploratory visualization of sample distribution among the classes, I plot the distribution histogram below. The sample of sign is not balanced, as some signs are more representative then others. However this representativenes proportion is kept among training, validation and testing sample. 

As the sample is not uniformly distributed abong the classes it is possible that for some classes we can achieve better prediction than for other classes. As proportions of signs among training, validation and testing samples are relatively constant, we can expect the performance of between training, validation and between training and testing not being biased by sample selection
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For data preprocessing I allow the option to to convert the images to grayscale. This reduces the input data by averaging the 3 channels (RGB) into one channel. However the validation performance is not neccessarily better, as colors in sign (especially red) have a significance. In the final model I choose, I leave the three channels data as input. The images are normalized by the conventional way of detracting and dividing each pixel x by 128: x1 = (x-128)/128. This can smooth the effect of extremely high RGB values.

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

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 





---


 
``` 

---
## Resources
* Udacity Self-Driving Car [Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) 
* Udacity project assignment and template on [GitHub](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)
* Udacity project [rubric](https://review.udacity.com/#!/rubrics/481/view)


