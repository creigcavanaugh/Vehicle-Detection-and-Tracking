from vdlib import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split


###
# Creig Cavanaugh - April 2017   (ver vd_b_17)

##
heatmap_visualization = True



###########
#Hyper parameters

#For Classifier
color_space = 'RGB' # Can be RGB, HSV, *LUV, HLS, YUV, YCrCb
spatial = 16
histbin = 32
orient = 12  # HOG orientations 12
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
hist_range=(0, 256)  #

#Additional for Search
ystart = 400
ystop = 656
#scale = 2.5
spatial_size = (spatial, spatial) # Spatial binning dimensions
hist_bins = histbin

heatboxset_frames = 7  #rcb 7
heat_thresh = 6   #rcb 6

heatboxset=[]

###############

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    heatboxes = []
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch = np.copy(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps+1):
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
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    

            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            #test_prediction = 1

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) #255, 6
                    
                x1 = int(xleft * scale)
                y1 = int(ytop * scale) + ystart
                x2 = int((xleft+window)*scale)
                y2 = int((ytop+window)*scale)+ystart
                #print ("x1: ", x1, "y1: ", y1, "x2: ", x2, "y2: ", y2 )
                heatboxes.append(((x1, y1), (x2, y2)))

    return draw_img, heatboxes



def process_image(image):
    #returned_img = pipeline(image, mtx, dist)
    out_img_1, heatboxes_1 = find_cars(image, ystart, ystop, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    out_img_2, heatboxes_2 = find_cars(image, ystart, ystop, 2.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    #out_img_3, heatboxes_3 = find_cars(image, ystart, (ystart + (64*2)), 1.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    out_img_3, heatboxes_3 = find_cars(image, ystart, ystop, 3.0, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # plt.imshow(out_img_1)
    # plt.show()

    # plt.imshow(out_img_2)
    # plt.show()

    #plt.imshow(out_img_3)
    #plt.show()

    heatboxes = heatboxes_1 + heatboxes_2 + heatboxes_3

    #heatboxes = heatboxes_1 + heatboxes_2

    heatboxset.append(heatboxes)

    if len(heatboxset) > heatboxset_frames:
        heatboxset.pop(0)

    heatboxes = []

    #print('heatbox len:', len(heatboxset))


    for xi in range(len(heatboxset)):
        heatboxes = heatboxes + heatboxset[xi]

    #print('hbx:', heatboxes)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,heatboxes)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,heat_thresh)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    #Save copy of heatmap for visualization
    if (heatmap_visualization == True):
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.savefig('output_images/heatmap17a.png')
        plt.close()

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    #Add in visuals 

    scaled_out_img = cv2.resize(out_img_2,None,fx=.3, fy=.3, interpolation = cv2.INTER_CUBIC)

    x_offset=20
    y_offset=20
    draw_img[y_offset:y_offset+scaled_out_img.shape[0], x_offset:x_offset+scaled_out_img.shape[1]] = scaled_out_img



    #Read in the heatmap, and superimpose over draw_img array
    if (heatmap_visualization == True):
        img_heat = cv2.imread('output_images/heatmap17a.png')
        scaled_heat = cv2.resize(img_heat ,None,fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC)
        
        x_offset=draw_img.shape[1]-(scaled_heat.shape[1]+20)
        y_offset=20
        draw_img[y_offset:y_offset+scaled_heat.shape[0], x_offset:x_offset+scaled_heat.shape[1]] = scaled_heat

    return draw_img
    #return out_img_1




###############
# Create Classifier

# Read in car and non-car images
cars, notcars = get_image_set(maximages=5000, verbose=True)



car_features = extract_features(cars, color_space=color_space, spatial_size=(spatial, spatial),
                        hist_bins=histbin, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, hog_channel=hog_channel, 
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(notcars, color_space=color_space, spatial_size=(spatial, spatial),
                        hist_bins=histbin, hist_range=hist_range, orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, hog_channel=hog_channel, 
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)


# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

#Normalize Features
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

#Use GridSearch to optimize hyperparameters
# from sklearn.grid_search import GridSearchCV
# from sklearn.svm import SVC
# parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 0.5, 1, 5, 10, 100], 'gamma': [0.001, 0.0001]}
# #svr = svm.SVC()
# svr = SVC()
# clf = GridSearchCV(svr, parameters)
# clf.fit(X_train, y_train)
# print (clf.best_params_)

#Linear SVC 
# svc = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0)

#GridSearch optimized parameters
svc = svm.SVC(C=1, kernel='rbf', gamma=0.0001)
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
print('Prediction Shape:', X_test[0].shape)


t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


# Test on image
# img = mpimg.imread('./test_images/test1.jpg')

# out_img, heatboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

# plt.imshow(out_img)
# plt.show()

# heat = np.zeros_like(img[:,:,0]).astype(np.float)

# # Add heat to each box in box list
# heat = add_heat(heat,heatboxes)
    
# # Apply threshold to help remove false positives
# heat = apply_threshold(heat,1)

# # Visualize the heatmap when displaying    
# heatmap = np.clip(heat, 0, 255)

# # Find final boxes from heatmap using label function
# labels = label(heatmap)
# draw_img = draw_labeled_bboxes(np.copy(img), labels)

# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(draw_img)
# plt.title('Car Positions')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='hot')
# plt.title('Heat Map')
# fig.tight_layout()
# plt.show()


#Build video
#clip1 = VideoFileClip("short2.mp4")
clip1 = VideoFileClip("project_video.mp4")
output = 'output_images/video_output.mp4'

clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)






