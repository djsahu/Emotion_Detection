# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:46:25 2020

@author: KIIT
"""

import os
import cv2
import numpy as np

data_dir= r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\emotion-detection-from-facial-expressions\kaggle_data\test"
lstdr=os.listdir(data_dir)

pathOut=r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\emotion-detection-from-facial-expressions\kaggle_data\test"


train_images = []
train_labels = []


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(300,300))
    return img


c=0
for i in lstdr[1:]:
   img = cv2.imread(os.path.join(data_dir,i))
   train_images.append(preprocess(img))
   cv2.imwrite(pathOut + "\\test%d.jpg" % c, preprocess(img))
   train_labels.append(i)
   c=c+1
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)



