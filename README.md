### Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/figure_1.png
[image2]: ./output_images/figure_2.png
[image3]: ./output_images/figure_3.png
[image4]: ./output_images/figure_4.png
[image5]: ./output_images/figure_5.png
[image6]: ./output_images/figure_6.png
[video1]: ./output_images/P5_video.mp4 

### Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/513/view) points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 
_Meets Specifications: The writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled._

You're reading it!

### Histogram of Oriented Gradients (HOG) 

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 18 through 117 of the file called `P5_code.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of five of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
_Meets Specifications: Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why._

I systematically conducted trial runs to compute the prediction accuracy using linear SVM (from lecture 29) as a function of various HOG parameters. The code for this step is contained in lines 119 through 246 of the file `P5_code.py`. The log of each run is shown in a separate file ('HOG_trials.csv').

First I extracted features of `Vehicles` and `Non-vehicles` using a function `extract_features` provided in the lecture. I modified the function to include HOG parameters, spatial color features and color histogram features. However, at this point I optimized only the HOG parameters, and left the spatial color features and color histogram features to the next section.

I used the smaple size of 500 (each class) to evaluate the accuracy using different color space, and found that `LUV` (accuracy = 1.0), `YUV` (accuracy = 0.995), and `YCrCb` (accuracy = 0.99) result in a relatively high accuracy. The HOG channel (= 0), orientation (= 9), pix_per_cell (= 8) and cell_per_block (= 2) were held constant.

Then I increased the sample size to 8,792 (each class). Given the data set, this is the maximum possible sample size with an equal number of samples in each class. I learned that the accuracy was slightly lower around ~0.95. To reduce the computation time, I also used the samlple size of 2,000 (each class), which produced the accuracy of 0.95-0.97.

Next I changed the HOG channel and found that `ALL` resulted in the highest accuracy (>0.99) with the color space `YUV` and `YCrCb`. I used the sample size to 8,792 (each class) and confirmed that the color space `YCrCb` with the HOG channel `ALL` resulted in the higher accuracy (= 0.9838) than the color space `YUV` with the HOG channel `ALL` (= 0.9821).

In a similar manner, I determined other HOG parameters. For pixels_per_cell, `(4, 4)` actually produced the highest accuracy but the computation time was significantly longer. Therefore I chose `(8, 8)` instead, which produced the second highest accuracy with a reasonable computation time. The final choice of HOG parameters are shown below: 

|HOG parameters| Tested values | Final value|
|:---------------------:|:---------------------------:|:---------------------------:| 
|Color space|'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'|'YCrCb'|
|HOG channel|0, 1, 2, 'ALL'|'ALL'|
|Orientation|8, 9, 10, 11, 12, 13|10|
|Pixels per cell|(4, 4), (8, 8), (16, 16), (32, 32)|(8, 8)|
|Cells per block|(1, 1),(2, 2),(3, 3),(4, 4)|(2, 2)|

The prediction accuracy with this combination of HOG parameters was 0.9841.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
_Meets Specifications: The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier._

Just like HOG parameters, I systematically conducted trial runs to compute the prediction accuracy using linear SVM (from lecture 29) as a function of color feature parameters. The code for this step is contained in lines 248 through 309 of the file `P5_code.py`. The log of each run is shown in a separate file ('Color_trials.csv').

The final choice of color parameters are shown below: 

|Color parameters| Tested values | Final value|
|:---------------------:|:---------------------------:|:---------------------------:| 
|Spatial color size|8,16,32,64|16|
|Color histogram bins|8,16,32,64|32|

The prediction accuracy with the optimized HOG and color feature parameters was 0.9909.

Next, I combined those features by `numpy.vstack()` where each row is a single feature vector. Then I normalized the feature vector by `StandardScaler()`. The resulting features are scaled to zero mean and unit variance. I split the data set (n=8,792 each class) into the training set (80%) and the test set (20%), and trained a classifier using a linear SVM `LinearSVC()`.

Finally, for optimization of SVM parameter 'C', I used `GridSearchCV`. 

|SVM parameter| Tested values | Final value|
|:---------------------:|:---------------------------:|:---------------------------:| 
|C|0.1,1,10|10|

The prediction accuracy with the optimized HOG parameters, color feature parameters, and C parameter was 0.9903. I don't quite understand why the accuracy went down from 0.9909 to 0.9903 with an optimized C value, but my speculation is that it is a cost of reducing overfitting.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
_Meets Specifications: A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen._

The code for this step is contained in lines 343 through 419 of the file `P5_code.py`. I used `find_cars` function provided in the lecture to implement a sliding window search. This function extracts HOG features once, which are sub-sampled to get all of the overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. After severl iterations I chose the scale of 1.5 that resulted in correct identification of cars. However, sliding window alone still leads to false-positives. One example is shown below.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
_Meets Specifications: Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)_

The code for this step is contained in lines 420 through 475 of the file `P5_code.py`. I used `add_heat`, `apply_threshold`, `draw_labeled_bboxes` functions provided in the lecture to reduce false positives. Some examples are shown below.

![alt text][image4]

In this example, a total of three cars are correctly identified. A car on the other side of the lane is also correctly identfied. No false-positives.

![alt text][image5]

In this example, cars are correctly not identified. No false-positives.

![alt text][image6]

In this example, a total of two cars are correctly identified. No false-positives.

---
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
_Meets Specifications: The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) on each frame of video._

Here's a [link to my video result](./output_images/P5_video.mp4 )

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
_Meets Specifications: A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur._

The code for this step is contained in lines 477 through 511 of the file `P5_code.py`. I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. 

The threshold of heatmap was applied not only spatially but also temporally to effectively 1) identify vehicles, which tend to stay in the ROI for multiple frames; and 2) remove false-positives, which tend to appear inconsistently from frame to frame.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
_Meets Specifications: Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail._

I learned that a high accuracy of SVM prediction is only a necessary but not a sufficient condition for the pipeline to be effective. For example, I was happy with the high accuracy (>0.99) of SVM prediction after optimization of HOG and color parameters. However, the pipeline still identifies false-positives. Overfitting is certainly one of the issues, and this could potentially be resolved with further SVM parameter optimization. Another consideration is the method of optimization of HOG and color parameters. In this project I optimized one parameter first, and moved on to the next parameter, while first parameter is held constant at the 'optimal' value. In other words, I did not conduct exhaustive optimization using every possible combination of HOG and color parameters. Such an exhaustive optimization may have led to a better combination of parameters.

