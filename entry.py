from train import TrainSVM
from detector import Detector
import cv2

# loading the train data, positive w. cars and negative wo. cars
pos_path = 'D:\Sabahuddin\camera\cars\pos_new'
neg_path = 'D:\Sabahuddin\camera\cars\_neg_new'
trainer = TrainSVM(pos_path, neg_path)
trainer.train()
cap = cv2.VideoCapture('D:\Sabahuddin\camera\Kamerasemafor3_01.avi')
#
Detector().find_cars(cap)
