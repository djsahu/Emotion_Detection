# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:20:08 2020

@author: KIIT
"""

import os
import cv2
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import Adam,RMSprop
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D,MaxPooling2D

data_dir = r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\KDEF_without_fl_fr\train"
lstdr=os.listdir(data_dir)

val_dir=r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\KDEF_without_fl_fr\test"
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

import pickle
with open('train_images_wo_tl_tr.p','wb') as f:
    pickle.dump(train_images,f)
with open('train_labels_wo_tl_tr.p','wb') as f:
    pickle.dump(train_labels,f)
with open('val_images_wo_tl_tr.p','wb') as f:
    pickle.dump(val_images,f)
with open('val_labels_wo_tl_tr.p','wb') as f:
    pickle.dump(val_labels,f)
    
import matplotlib.pyplot as plt
plt.imshow(train_images[0],cmap='gray')
plt.title(train_labels[0],color='b')
plt.show()

train_images.shape

le=LabelEncoder()
train_labels = le.fit_transform(train_labels)
val_labels = le.fit_transform(val_labels)

n_classes = len(set(train_labels))
n_cols=5

fig, axes= plt.subplots(n_classes, n_cols, figsize=(15,15))
fig.tight_layout() #seperate the images

for i in range(n_cols):
    for j in range(n_classes):
        selected_image= random.choice(train_images[train_labels==j])
        axes[j][i].imshow(selected_image,cmap='gray')
        
n_pixels=train_images.shape[1]*train_images.shape[2]
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(n_pixels)
    img=img/255 #scaling 
    return img

train_images= np.array(list(map(preprocess,train_images)))
val_images= np.array(list(map(preprocess,val_images)))

train_images = train_images.reshape(train_images.shape[0],48,48,1)/255
val_images = val_images.reshape(val_images.shape[0],48,48,1)/255
train_images.shape

train_labels = to_categorical(train_labels,n_classes)
val_labels = to_categorical(val_labels,n_classes)

model= Sequential()

model.add(Conv2D(16,(3,3),input_shape=(48,48,1),activation='relu'))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes,activation='softmax'))
model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

h=model.fit(train_images,train_labels,epochs=5,verbose=1,validation_data=(val_images,val_labels))