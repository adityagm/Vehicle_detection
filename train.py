import numpy as np
from save_model import save_model, save_feature_data
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from feature_extraction import extract_features
import time
import os
import cv2
import pickle


class TrainSVM:
    # orientations, pixels_per_cell, cells_per_block, visualize
    def __init__(self, pos_path, neg_path):
        self.config = {'orientations': 9, 'pixels_per_cell': 8, 'cells_per_block': 2, 'visualize': False}

        self.pos_path = pos_path
        self.neg_path = neg_path

    def train(self):
        pos_features = []
        neg_features = []
        print(self.pos_path)
        print(self.neg_path)
        # pos_files = [os.path.join(rootdir, file) for rootdir, _, files in os.walk(self.pos_path) for file in files]
        # neg_files = [os.path.join(rootdir, file) for rootdir, _, files in os.walk(self.neg_path) for file in files]

        for file in os.listdir(self.pos_path):
            path = os.path.join(self.pos_path, file)
            image = cv2.imread(path)
            pos_features.append(extract_features(image, self.config))

        for file in os.listdir(self.neg_path):
            path = os.path.join(self.neg_path, file)
            image = cv2.imread(path)
            neg_features.append(extract_features(image, self.config))


            # random shuffling of the features
        random.shuffle(pos_features)
        random.shuffle(neg_features)

        print("{} positive features, {} negative features \n".format(len(pos_features), len(neg_features)))

        print("scaling.... \n")

        xscaler = StandardScaler().fit(pos_features + neg_features)
        pos_features = xscaler.transform(pos_features)
        neg_features = xscaler.transform(neg_features)

        print("Saving features to file Features \n")
        file = "D:\Sabahuddin\svm_hog_speed\FEATURE_DATA.p"
        try:
            pickle.dump({"positive": pos_features, "negative": neg_features}, open(file, 'wb'))
            print("Feature Data saved to {}".format(file))
        except Exception as e:
            print('Failed to save the model at the destination file {}:{}'.format(file, e))
            raise

        features = np.vstack((pos_features, neg_features)).astype(float)
        labels = np.hstack((np.ones(len(pos_features)), np.zeros(len(neg_features))))

        print(" splitting the features into train and validation sets... \n")
        xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3, random_state=42)

        print(" size of train set {}".format(len(xtrain)))
        print(" size of test set {}".format(len(xtest)))

        svm = LinearSVC(max_iter=3000, C=1, loss="squared_hinge", penalty='l1', dual=False, fit_intercept=False)
        start_time = time.time()
        print(" training the classifier with the train set... \n")
        svm.fit(xtrain, ytrain)
        print(" trained in {:.1f}s".format(time.time() - start_time))
        # ytest = ytest.reshape(1, -1)
        prediction = svm.predict(xtest)
        print("prediction \n", prediction)
        print("ytest \n", ytest)
        print("validation accuracy is {:f}".format(svm.score(xtest, ytest)))

        # clf_model, scaler, file, config
        save_model(svm, xscaler, 'D:\Sabahuddin\svm_hog_speed\MODEL_SVM_HOG_try1.p', self.config)
