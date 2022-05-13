import sys
import os.path as path
import glob
import cv2 as cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#Parameters
orient = 12  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat= True
hist_feat= True
hog_feat=True


def spatial_features(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),transform_sqrt=False,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),transform_sqrt=False,
                       visualize=vis, feature_vector=feature_vec)
        return features


def Single_extract_features(image_orig, spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256),
                        orient=9,pix_per_cell=8, cell_per_block=2,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    images=[image_orig]
    image_flipped= cv2.flip(image_orig,1)
    images.append(image_flipped)

    for image in images:
        feature_image=None

        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        image_feature=[]
        if spatial_feat == True:
            image_feature= spatial_features(feature_image, spatial_size)

        histogram_features=[]
        if hist_feat == True:
            histogram_features= color_hist(feature_image, hist_bins, hist_range)

        if hog_feat == True:
            hog_features = []
            hog_features = get_hog_features(feature_image[:,:,0],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True)
            hog_features = np.ravel(hog_features)
            features = np.concatenate((image_feature, histogram_features,hog_features))

    return features


def extract_features(imgs, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
                        orient=9,pix_per_cell=8, cell_per_block=2,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    print('extraction started')
    for image_orig in imgs:
        single_feature = Single_extract_features(image_orig, spatial_size, hist_bins, hist_range, orient, pix_per_cell,
                                                 cell_per_block, spatial_feat, hist_feat, hog_feat)
        features.append(single_feature)
    print('extraction ended')
    return features


# if svm classifer exist, load it; otherwise, compute the svm classifier
clf_path = 'clf_pickle_all.p'
if path.isfile(clf_path):
    print('loading existing classifier...')
    with open(clf_path, 'rb') as file:
        clf_pickle = pickle.load(file)
        clf = clf_pickle["clf"]
        scaler = clf_pickle["scaler"]
        orient = clf_pickle["orient"]
        pix_per_cell = clf_pickle["pix_per_cell"]
        cell_per_block = clf_pickle["cell_per_block"]
        spatial_size = clf_pickle["spatial_size"]
        hist_bins = clf_pickle["hist_bins"]

else:
    # reading image paths with glob
    vehicle_image_arr = glob.glob('./vehicles/*/*.png')

    # read images and append to list
    vehicle_images_original = []
    for imagePath in vehicle_image_arr:
        readImage = cv2.imread(imagePath)
        rgbImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
        vehicle_images_original.append(rgbImage)

    print('Reading of Vehicle Images Done')

    non_vehicle_image_arr = glob.glob('./non-vehicles/*/*.png')

    non_vehicle_images_original = []
    for imagePath in non_vehicle_image_arr:
        readImage = cv2.imread(imagePath)
        rgbImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
        non_vehicle_images_original.append(rgbImage)

    print("Reading of Non Vehicle Images Done")

    print("No. of Vehicle Images Loaded " + str(len(vehicle_image_arr)))
    print("No. of Non-Vehicle Images Loaded " + str(len(non_vehicle_images_original)))

    vehicleFeatures = extract_features(vehicle_images_original, spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    nonVehicleFeatures = extract_features(non_vehicle_images_original, spatial_size=spatial_size, hist_bins=hist_bins,
                                          orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    FeaturesList = np.vstack([vehicleFeatures, nonVehicleFeatures]).astype(np.float64)
    LabelList = np.hstack([np.ones(len(vehicleFeatures)), np.zeros(len(nonVehicleFeatures))])

    X_train, X_test, Y_train, Y_test = train_test_split(FeaturesList, LabelList, test_size=0.2, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    print('Training started')
    clf = LinearSVC()
    clf.fit(X_train, Y_train)
    print("Accuracy of SVC is  ", clf.score(X_test, Y_test))

    # save classifier
    clf_pickle = {}
    clf_pickle["clf"] = clf
    clf_pickle["scaler"] = scaler
    clf_pickle["orient"] = orient
    clf_pickle["pix_per_cell"] = pix_per_cell
    clf_pickle["cell_per_block"] = cell_per_block
    clf_pickle["spatial_size"] = spatial_size
    clf_pickle["hist_bins"] = hist_bins

    destnation = clf_path
    pickle.dump(clf_pickle, open(destnation, "wb"))
    print("Classifier is written into: {}".format(destnation))




