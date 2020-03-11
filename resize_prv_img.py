# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:46:25 2020

@author: KIIT
"""

import os
import cv2
import numpy as np
import seaborn as sns
import random

data_dir= r"C:\Users\KIIT\Downloads\Compressed\KDEF_and_AKDEF\KDEF_and_AKDEF\KDEF"
lstdr=os.listdir(data_dir)

pathOut=r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\New folder"


train_images = []
train_labels = []


def preprocess(img):
    img = cv2.resize(img,(300,300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

c=0
for i in lstdr:
   img_dir = os.path.join(data_dir,i)
   
   for j in os.listdir(img_dir):
       img = cv2.imread(os.path.join(img_dir,j))
       train_images.append(preprocess(img))
       cv2.imwrite(pathOut + "\\new%d.jpg" % c, preprocess(img))
       train_labels.append(i)
       c=c+1
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)



import matplotlib.pyplot as plt
plt.imshow(train_images[0],cmap='gray')
plt.title(train_labels[0],color='b')
plt.show()

import matplotlib.pyplot as plt
n_classes = len(set(train_labels))
n_cols=7

fig, axes= plt.subplots(n_classes, n_cols, figsize=(15,15))
fig.tight_layout() #seperate the images

for i in range(n_cols):
  for j in range(n_classes):
    selected_image= random.choice(train_images[train_labels==j])
    axes[j][i].imshow(selected_image,cmap='gray')


