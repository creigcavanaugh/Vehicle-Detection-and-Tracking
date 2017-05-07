
## Vehicle Detection Project

**Creig Cavanaugh - April 2017**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/hog_channels.png
[image2b]: ./output_images/spatial_color_histogram.png
[image3a]: ./output_images/scale_15.png
[image3b]: ./output_images/scale_25.png
[image3c]: ./output_images/scale_30.png
[image4a]: ./output_images/not_detecting_side_car.png
[image4b]: ./output_images/Window_Search_example.png
[image5a]: ./output_images/heat_map_car_positions_example.png
[image5b]: ./output_images/video_preview.png

[video1]: ./video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 226 through 235 of the file called `vehicle_detection.py`, which calls the `extract_features` function
in the module `vdlib.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters  - I seemed to have the best results staying with 9 pixels per cell, and two cells per block.  I experimented with all the color spaces, and interestingly RGB works the best with the pipeline I setup, which uses all color channels for HOG. 

I also experimented with using different HOG orientations - I got good results with 7 orientations using the SVM linear kernel, but switched to the RBF kernel and ended up using 12.

I also included both Spatial and Color Histogram into the feature set.  I experimented initially with just using the HOG features, but got more robust results when adding in the additional features.

![alt text][image2b]

Here are my final parameters:
```color_space = 'RGB'
spatial = 16
histbin = 32
orient = 12  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).


I created a function called `get_image_set` in the `vdlib.pl` module that provides randomly shuffled image filenames for use in training the classifier.  I ended up using 5000 vehicle and non-vehicle images to train the classifier.  The function is called on line 222 in my code in `vehicle_detection.py`.

I used the `extract_features` function in the `vdlib.pl` module to extract the HOG and color features from the training data.  Features were labeled 1=car and 0=not-car.  The features were normalized using the `StandardScaler()` from the `sklearn.preprocessing` package.  On line 252 in `vehicle_detection.py`, the data set is randomized and split between training and test data (80% / 20%).  

I used GridSearchCV to determine optimized hyper-parameters, using the following command:
```
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 0.5, 1, 5, 10, 100],'gamma': [0.001, 0.0001]}
svr = SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
print (clf.best_params_)

```

The output of GridSearch indicated RBF is the optimal kernel, using C=1.0 and Gamma=0.0001.  The SVM training is done on lines 276 through 289 in `vehicle_detection.py`.



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented this step in lines 56 through 130 in my code in `vehicle_detection.py` in the function called `find_cars`.  I utilized the sliding window approach as described in the 'Hog Sub-sampling Window Search' section of the Vehicle Detection and Tracking lesson. I limited the HOG search window, using ystart = 400 and ystop = 656. I experimented with different scales to find the best detection performance, and ended up calling the function three times to search at the following scales: 1.5, 2.5 and 3.0

Here are examples of the grid pattern for each: 

**1.5 Scale**
![alt text][image3a]

**2.5 Scale**
![alt text][image3b]

**3.0 Scale**
![alt text][image3c]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:


Here is an early result, which does not pick up both cars:
![alt text][image4a]

Here is an improved result:


![alt text][image4b]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I implemented this step in lines 134 through 191 in my code in `vehicle_detection.py`.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

![alt text][image5a]

In addition, I added the positions of positive detections to an array that holds the previous 7 video frames, and that compilation of detections is used to generate the heatmap, which helps to further reduce false positives.  Because of this, I use a heat threshold of 6.

The video result provides a active view of the heat map in the upper right of the video.

![alt text][image5b]




---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I had some initial difficulty with importing jpg vs png files, and the nuances of handling them differently in the code.  Similarly, need to be careful with the RGB vs BGR differences.   

I initially used the linear kernel in SVM, and switched to the RBF kernel after using GridSearch and finding RBF performed the best.  I noticed I had to re-tune some of my other parameters after the switch to RBF, and also noticed processing took longer when implementing RBF.  A future improvement could be to improve and simplify the extracted features - this could minimize the performance difference between the two kernels, and make it practical to possibly switch back to linear SVM for quicker processing times.

The data sets used were of 64x64 images of vehicles and non-vehicles.  Many of the images in the dataset appear to be smaller hatchback type cars, which is not exactly representative of vehicles in the US. Additional types of images, such as broader categories of vehicles and more side views could help performance.  I would also further experiment using higher resolution data-set of vehicles to see if that has any impact to vehicle detection performance.

During the beginning of the tuning process, I noticed there were a lot of false positives when other structural components were in view, specifically structures like the guard rail on the bridges.  This could be an indication the pipeline could fail in more urban settings with structures such as buildings, fences, signs, bridges and tunnels. To possibly help, more of these types of images could be added to the "non-vehicle" image set.

In addition, night-time and adverse weather conditions would probably greatly reduce vehicle detection performance in its current state.  A future improvement could be to develop a classifier tuned for nighttime conditions or bad weather. 
