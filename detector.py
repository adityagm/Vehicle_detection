from save_model import load_model
from sliding_windows import sliding_windows
import numpy as np
from collections import deque
from feature_extraction import extract_features, extract_hog_features
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label
import cv2


class Detector:

    def __init__(self, init_size=(400, 400), x_overlap=0.0, y_overlap=0.05, x_range=(0, 1), y_range=(0.4, 1),
                 scale=1.0):
        self.init_size = init_size
        self.x_overlap = x_overlap
        self.y_step = y_overlap
        self.x_range = x_range
        self.y_range = y_range
        self.scale = scale
        self.windows = None

        # loading the pickled model from the file as classifier
        clf = load_model('MODEL_SVM_HOG_try1.p')
        self.svc = clf['classifier']
        self.scaler = clf['scaler']
        self.orientations = clf['orientations']
        self.pixels_per_cell = clf['pixels_per_cell']
        self.cells_per_block = clf['cells_per_block']
        self.visualize = clf['visualize']
        self.config = {'orientations': self.orientations, 'pixels_per_cell': self.pixels_per_cell,
                       'cells_per_block': self.cells_per_block, 'visualize': self.visualize}

    def classify(self, image):

        feature_vector = [extract_features(image[y_upper:y_lower, x_upper:x_lower, :], self.config)
                          for (x_upper, y_upper, x_lower, y_lower) in self.windows]

        # print("feature Vector \n", feature_vector)
        # removing features that have none in them
        # for i in range(len(feature_vector)):
        #     if feature_vector[i] is not None:
        #         feature_vector_no_none.append(feature_vector[i])
        # print(feature_vector_no_none)
        feature_vector = self.scaler.transform(feature_vector)
        predictions = self.svc.predict(feature_vector)

        return [self.windows[id] for id in np.argwhere(predictions == 1)[:, 0]]

    def find_cars(self, video, num_of_frames=9, threshold=120, min_bbox=None, draw_heatmap=True, draw_heatmap_size=0.25,
                  vid_write=True, vid_write_fps=24):

        cap = video
        if not cap.isOpened():
            raise RuntimeError("error in opening Video file")

        grab, frame = cap.read()
        height, width = frame.shape[:2]

        # generating the windows and storing the frame for further classification
        # getting the sliding windows
        # sliding_windows(image_size, init_size, x_overlap, y_step, x_range, y_range, scale)
        # just need to generate the sliding window coords once, it remains static throughout the video
        self.windows = sliding_windows((width, height), self.init_size, self.x_overlap, self.y_step, self.x_range,
                                       self.y_range, self.scale)
        if min_bbox is None:
            min_bbox = (int(0.1 * width), int(0.1 * height))
        '''for testing purposes only'''
        (x, y, z) = (0,0,0)
        for (x_upper, y_upper, x_lower, y_lower) in self.windows:
            x += 30
            y = 255
            z += 30
            cv2.rectangle(frame, (x_upper, y_upper), (x_lower, y_lower), (x, y, z), 2)
        # # # cv2.imshow(" ", frame)
        plt.imshow(frame, interpolation='bicubic')
        plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()'''
        # setting the size of the heatmap to be generated
        inset_heatmap_size = (int(draw_heatmap_size * width), int(draw_heatmap_size * height))

        # initialize the current heat map for each frame analyzed
        curr_heat_map = np.zeros((frame.shape[:2]), dtype=np.uint8)
        aggregate_heat_map = np.zeros_like(curr_heat_map, dtype=np.uint8)
        last_N_frames = deque(maxlen=num_of_frames)
        heatmap_labels = np.zeros_like(curr_heat_map, dtype=np.uint8)

        # generating the weights for aggregating the heatmaps since the the most recent heatmap must be given weighted higher
        # and the earliest the least
        weight = np.linspace(1 / (num_of_frames + 1), 1, num_of_frames)

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            curr_heat_map[:] = 0
            aggregate_heat_map[:] = 0

            for (x_upper, y_upper, x_lower, y_lower) in self.classify(frame):
                curr_heat_map[y_upper:y_lower, x_upper:x_lower] += 10

            last_N_frames.append(curr_heat_map)

            for i, heatmap in enumerate(last_N_frames):
                cv2.add((curr_heat_map * weight[i]).astype(np.uint8), aggregate_heat_map, dst=aggregate_heat_map)

            cv2.dilate(aggregate_heat_map, np.ones((7, 7), dtype=np.uint8), aggregate_heat_map)

            if draw_heatmap:
                inset = cv2.resize(aggregate_heat_map, inset_heatmap_size, interpolation=cv2.INTER_AREA)
                inset = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
                frame[:inset_heatmap_size[1], :inset_heatmap_size[0], :] = inset

            # thresholding, ignoring the pixels that are <= threshold
            aggregate_heat_map[aggregate_heat_map <= threshold] = 0

            # labeling the blobs
            num_objects = label(aggregate_heat_map, output=heatmap_labels)
            print("number of objects", num_objects)
            # locating the largest bounding box around each box
            for obj in range(1, num_objects + 1):
                (y_coord, x_coord) = np.nonzero(heatmap_labels == obj)
                x_upper, y_upper = min(x_coord), min(y_coord)
                x_lower, y_lower = max(x_coord), max(y_coord)

                # drawing the bounding box if it is bigger that the minimum box
                if (x_lower - x_upper > min_bbox[0] and y_lower - y_upper > min_bbox[1]):
                    cv2.rectangle(frame, (x_upper, y_upper), (x_lower, y_lower), (255, 255, 0), 2)

            if vid_write:
                # plt.imshow(frame, interpolation='bicubic')
                # plt.show()
                cv2.imshow(" ", frame)
                cv2.waitKey(1)

        cap.release()
