import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
from scipy.ndimage.measurements import label
import random
import time
from moviepy.editor import VideoFileClip

## Read in cars (carY) and notcars (carN)
carY = glob.glob('test_images/vehicles/*/*.png')
carN = glob.glob('test_images/non-vehicles/*/*.png')

# Show 5 random images for each class
fig, axs = plt.subplots(2,5, figsize=(15,6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(0,5):
    id_carY = random.randint(0,len(carY))
    img_carY = mpimg.imread(carY[id_carY])
    axs[i].axis('off')
    axs[i].imshow(img_carY)
    axs[i].set_title('Vehicle',fontsize=20)
for i in range(5,10):
    id_carN = random.randint(0,len(carN))
    img_carN = mpimg.imread(carN[id_carN])
    axs[i].axis('off')
    axs[i].imshow(img_carN)
    axs[i].set_title('Non-vehicle',fontsize=20)

# HOG
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Show 1 random image for each class
orient = 9
pix_per_cell = 8
cell_per_block = 2

fig, axs = plt.subplots(2,6, figsize=(15,6))
fig.subplots_adjust(hspace = .001, wspace=.01)
axs = axs.ravel()

id = random.randint(0,len(carY))
img = mpimg.imread(carY[id])
img = convert_color(img, conv='RGB2YCrCb')
c1,c2,c3 = cv2.split(img)
axs[0].set_ylabel('Vehicle',fontsize=15)
axs[0].imshow(c1,cmap='gray')
axs[0].set_title('Channel 1',fontsize=15)
features, hog_image = get_hog_features(c1,orient,pix_per_cell,cell_per_block,vis=True,feature_vec=True)
axs[1].axis('off')
axs[1].imshow(hog_image,cmap='gray')
axs[1].set_title('HOG(1)',fontsize=15)
axs[2].axis('off')
axs[2].imshow(c2,cmap='gray')
axs[2].set_title('Channel 2',fontsize=15)
features, hog_image = get_hog_features(c2,orient,pix_per_cell,cell_per_block,vis=True,feature_vec=True)
axs[3].axis('off')
axs[3].imshow(hog_image,cmap='gray')
axs[3].set_title('HOG(2)',fontsize=15)
axs[4].axis('off')
axs[4].imshow(c3,cmap='gray')
axs[4].set_title('Channel 3',fontsize=15)
features, hog_image = get_hog_features(c3,orient,pix_per_cell,cell_per_block,vis=True,feature_vec=True)
axs[5].axis('off')
axs[5].imshow(hog_image,cmap='gray')
axs[5].set_title('HOG(3)',fontsize=15)

id = random.randint(0,len(carN))
img = mpimg.imread(carN[id])
img = convert_color(img, conv='RGB2YCrCb')
c1,c2,c3 = cv2.split(img)
axs[6].set_ylabel('Non-vehicle',fontsize=15)
axs[6].imshow(c1,cmap='gray')
features, hog_image = get_hog_features(c1,orient,pix_per_cell,cell_per_block,vis=True,feature_vec=True)
axs[7].axis('off')
axs[7].imshow(hog_image,cmap='gray')
axs[8].axis('off')
axs[8].imshow(c2,cmap='gray')
features, hog_image = get_hog_features(c2,orient,pix_per_cell,cell_per_block,vis=True,feature_vec=True)
axs[9].axis('off')
axs[9].imshow(hog_image,cmap='gray')
axs[10].axis('off')
axs[10].imshow(c3,cmap='gray')
features, hog_image = get_hog_features(c3,orient,pix_per_cell,cell_per_block,vis=True,feature_vec=True)
axs[11].axis('off')
axs[11].imshow(hog_image,cmap='gray')

## HOG parameter optimization

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
sample_size = 8792
cars = carY[0:sample_size]
notcars = carN[0:sample_size]

# Tweak parameters and see how the results change
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
orient = 10
pix_per_cell = 8
cell_per_block = 2

t=time.time()
car_features = extract_features(cars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, hog_feat=True)
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, hog_feat=True)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

## Spatial color features and color histogram features optimization
sample_size = 8792
cars = carY[0:sample_size]
notcars = carN[0:sample_size]

# HOG paratemers are held constant
colorspace = 'YCrCb' 
hog_channel = 'ALL' 
orient = 10
pix_per_cell = 8
cell_per_block = 2

# Tweak color parameters 
spatial = 16
histbin = 32

car_features = extract_features(cars,cspace=colorspace,orient=orient, 
                                pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
                                hog_channel=hog_channel,spatial_size=(spatial,spatial),
                                hist_bins=histbin,
                                spatial_feat=True,hist_feat=True,hog_feat=True)

notcar_features = extract_features(notcars,cspace=colorspace,orient=orient, 
                                pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
                                hog_channel=hog_channel,spatial_size=(spatial,spatial),
                                hist_bins=histbin,
                                spatial_feat=True,hist_feat=True,hog_feat=True)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

## SVM parameter tuning
parameters = {'C':[0.1,1,10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(scaled_X, y) # clf.best_params_ = {'C': 10}

# Repeat with C=10
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using spatial binning of:',spatial,
    'and', histbin,'histogram bins')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC(C=10)
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

## Sliding windows

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img,ystart,ystop,scale,svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial,histbin):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    #nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    bbox_list = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg,size=(spatial,spatial))
            hist_features = color_hist(subimg, nbins=histbin)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                bbox_list.append(((xbox_left, ytop_draw + ystart),(xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return draw_img,bbox_list

img = mpimg.imread('test_images/test5.jpg')
ystart = 360
ystop = 656
scale = 1.5
    
draw_img,bbox_list = find_cars(img,ystart,ystop,scale,svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial,histbin)

plt.imshow(draw_img)

## heat map
heat = np.zeros_like(img[:,:,0]).astype(np.float)

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
    # Return thresholded map
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
    # Return the image
    return img

# Add heat to each box in box list
heat = add_heat(heat,bbox_list)
    
# Apply threshold to help remove false positives
heat = apply_threshold(heat,1)

# Visualize the heatmap when displaying    
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(img), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()

## Video pipeline
def image_pipeline(img):
    global boxes,ystart,ystop,scale,svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial,histbin
    # find cars
    draw_img,bbox_list = find_cars(img,ystart,ystop,scale,svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial,histbin)
    # Add bbox_list of the current frame to boxes
    smoothing = 22
    boxes.append(bbox_list)
    if len(boxes) < smoothing:
        thresh = len(boxes) - 1 
    else:
        thresh = smoothing    
    # Add heat to each box in box list
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    for boxlist in boxes[-smoothing:]:
        heat = add_heat(heat,boxlist)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,thresh)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat,0,255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img),labels)
    return draw_img


boxes = []
scale = 1.5
t0 = time.time()
output = 'P5_video.mp4'
clip1 = VideoFileClip('project_video.mp4')
output_clip = clip1.fl_image(image_pipeline)
output_clip.write_videofile(output,audio=False)
t1 = time.time()
print(round(t1-t0,2), 'sec to create a movie')