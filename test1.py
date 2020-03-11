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

data_dir = r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\face-expression-recognition-dataset\images\images\train"
lstdr=os.listdir(data_dir)

val_dir=r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\face-expression-recognition-dataset\images\images\validation"
lstdr_val=os.listdir(val_dir)

train_images = []
train_labels = []

val_images=[]
val_labels=[]


for i in lstdr:
    img_dir = os.path.join(data_dir,i)
    
    for j in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir,j))
        train_images.append(img)
        train_labels.append(i)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

for k in lstdr_val:
    val_img_dir = os.path.join(val_dir,k)
    
    for l in os.listdir(val_img_dir):
        val_img = cv2.imread(os.path.join(val_img_dir,l))
        val_images.append(val_img)
        val_labels.append(k)

val_images = np.array(val_images)
val_labels = np.array(val_labels)


import matplotlib.pyplot as plt
plt.imshow(train_images[0],cmap='gray')
plt.title(train_labels[0],color='b')
plt.show()

import matplotlib.pyplot as plt
plt.imshow(val_images[0],cmap='gray')
plt.title(val_labels[0],color='b')
plt.show()


n_classes = len(set(train_labels))
n_cols=5

fig, axes= plt.subplots(n_classes, n_cols, figsize=(15,15))
fig.tight_layout() #seperate the images

for i in range(n_cols):
  for j in range(n_classes):
    selected_image= random.choice(train_images[train_labels==j])
    axes[j][i].imshow(selected_image,cmap='gray')

sns.countplot(val_labels)
plt.show()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train_labels = le.fit_transform(train_labels)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
val_labels = le.fit_transform(val_labels)

train_images = train_images.reshape(train_images.shape[0],28,28,1)/255
val_images = val_images.reshape(val_images.shape[0],28,28,1)/255


