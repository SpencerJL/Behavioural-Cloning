#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* images of histogram of steering angles before and after data processing

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

![alt text][model summary procedure]

My model consists of a convolution neural network with 3x3 filter sizes and depths from 32 to 256 which consist 4 convolutional layers and followed by 3 fully connected layers. 

The model includes RELU layers to introduce nonlinearity (model.py line 116 - 124), and the data is normalized in the model using a Keras lambda layer (model.py line 114). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 118-121). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 195). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 207).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also the anti-clockwise to obtain more right steering angles. Meanwhile, I used all three cameras on the car during training to make sure the car can recover from off-center conditions.   

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet with 4 epochs I thought this model might be appropriate. However, the car went stright forward to the lake without any steering angle change which means I need some data processing. I tried a new lambda layer with x/127.5 - 1 instead of x/255 - 0.5, but the car still cannot pass the first turn. Moreover, I added the image cropping of (70,25), not much difference with last one. 

So I intriduce the VGG model instead of LeNet which added a single layer at the end of the model as it required. With the dropout and connection layers the train and validation loss were both quite low with augmented data and three cameras. The car can pass the first turn and worked well on most parts of road. However, the car went on the wrong track as shown below.

![alt text][wrong track]

As shown in the figure, the car directly went right onto the yellow road instead of the track. More data were needed especially at this turning corner. After this, the car run normally for the track completely.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes as shown in the figure.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][model summary procedure]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_counter_clockwise]

Then I record another anti-clockwise laps on track using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from the off-center. 

![alt text][left_back_to_center]

Then I repeated these three steps described above to obtain enough data.

To augment the data sat, I also flipped images and angles thinking that this would learn both left and right steering angles properly although I already recorded anti-clockwise. 

The data obtained from recording were not balanced as most of the steering angles were 0 as shown in figure.

![alt text][origin histogram and modified histogram]

The data were balanced by reducing the number of high bins especially the value of 0. 

![alt text][final histogram]

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the train and validation loss was stablelized. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The batch-size for model was set as 6 instead of 32 as there was a storage problem of GPU even on AWS. 
