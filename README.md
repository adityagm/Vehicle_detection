# Vehicle_detection
Entry->[trainSVMFile->extractFeatures[pos+neg]->trainSVM->saveModel]->detector[loadSavedModel->generateSlidingWindows->extractFeatures->standardize->predict->drawBoxes Based On Prediction With Sliding Windows As Reference]->save the coords-> ID generation and tracking.

# HOG-based linear SVM vehicle detection

The files in this repo are a framework I developed for training and utilizing a HOG-based linear SVM to detect vehicles and classify them (LMV or HMV) in a video. 

# Dataset

I have created a custome dataset with vehicle images extracted from the videos captured at two junctions in the country of Bosnia.

![](images/2020-10-08%20(2).png)

# Project Overview

# Objective

    Extract features from labeled (positive and negative) sample data, split into training and test sets, and finally Train classifier.
    For feature extraction, convert images to grey scale, select the desired channels, then extract HOG, color histogram, and spatial features.
    Detect and draw bounding boxes around objects in a video using a sliding window and smoothed heatmap, and also ID the vehicle detected.
    
   ![](images/2020-08-06%20(5).png)


