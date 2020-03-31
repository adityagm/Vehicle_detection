import cv2
import os
from skimage.feature import hog
import numpy as np


def extract_hog_features(image, orientations, pixels_per_cell, cells_per_block,
                         visualize=False, transform_sqrt=True, feature_vec=True, multichannel=None):

    if (visualize):

        features, hog_image = hog(image, orientations, (pixels_per_cell, pixels_per_cell),
                                  (cells_per_block, cells_per_block),
                                  transform_sqrt, visualize=True)

        return np.expand_dims(features, 1), hog_image

    else:
        multichannel = len(image.shape) > 2
        features = hog(image, orientations, (pixels_per_cell, pixels_per_cell), (cells_per_block, cells_per_block),
                       block_norm='L2-Hys', transform_sqrt=True, multichannel=multichannel, feature_vector=True)
        return np.expand_dims(features, 1)


def extract_features(feature_img, config):

    orientations = config['orientations']
    pixels_per_cell = config['pixels_per_cell']
    cells_per_block = config['cells_per_block']
    visualize = config['visualize']
    # print(pixels_per_cell, cells_per_block)
    hog_features = np.array([])

    feature_img = cv2.resize(feature_img, (64, 64))
    feature_img = cv2.cvtColor(feature_img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow(" ", feature_img)
    # cv2.waitKey(0)
    # print(feature_img.shape)
    if visualize:
        h_features, hog_img = extract_hog_features(feature_img, orientations, pixels_per_cell,
                                                        cells_per_block, visualize)

        hog_features = np.hstack((hog_features, h_features[:, 0])), hog_img

    else:
        feature_vector = extract_hog_features(feature_img, orientations, pixels_per_cell,
                                                     cells_per_block, visualize)
        hog_features = np.hstack((hog_features, feature_vector[:, 0]))

    return hog_features

