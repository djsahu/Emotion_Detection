# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 00:40:58 2020

@author: KIIT
"""

import os
import cv2
import numpy as np

data_dir= r"C:\Users\KIIT\Downloads\Compressed\KDEF_and_AKDEF\KDEF_and_AKDEF\KDEF"
lstdr=os.listdir(data_dir)

pathOut=r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\KDEF_with_fl_fr\test"


train_images = []
train_labels = []


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(256,256))
    return img

for i in lstdr:
   img_dir = os.path.join(data_dir,i)
   
   for j in os.listdir(img_dir):
       img = cv2.imread(os.path.join(img_dir,j))
       train_images.append(preprocess(img))
       if(j[4:6]=='AN'):
           cv2.imwrite(pathOut+"\\angry"+"\\"+ str(j[:8])+".jpg",preprocess(img))
       train_labels.append(i)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

#if((j[4:6]=='DI'): and j[6:8]!='FR') and (j[4:6]=='DI' and j[6:8]!='FL')):