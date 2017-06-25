import cv2
import numpy as np
import sys

img=cv2.imread('/home/fateh/PycharmProjects/keras/macular_hole/image_experiments/median_blur.jpg',0)
img=cv2.resize(img,(img.shape[1]/4,img.shape[0]/4))

