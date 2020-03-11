# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:47:58 2020

@author: cttc
"""

import cv2

cap= cv2.VideoCapture(0)

face_data= r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\data\haarcascades\haarcascade_frontalface_default.xml"

pathOut=r"C:\Users\KIIT\Pictures\Criterion Games"

face_images=[]

classifier = cv2.CascadeClassifier(face_data)

ret=True
c=0
while(ret):
    if cap.isOpened():
        ret,frame=cap.read()
        faces = classifier.detectMultiScale(frame,2,2)
        
        for x,y,w,h in faces:
            if(c<50):
                face_img= frame[y:y+h,x:x+w]
                face_images.append(face_img)
                cv2.imwrite(pathOut + "\\frame%d.jpg" % c, face_img)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
                c=c+1
        
        cv2.imshow('video',frame)
        if cv2.waitKey(24)==ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()

import numpy as np
import pickle

print(face_images)

face_images= np.array(face_images)

with open(r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\New folder\faces.p",'wb') as f:
    pickle.dump(face_data,f) 
    

with open(r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\New folder\faces.p", 'rb') as f:
  face_images = pickle.load(f)

import numpy as np
face_images=np.array(face_images)
shape=np.array(list(face_images.shape))[0]

import matplotlib.pyplot as plt
for i in range(shape):
    plt.imshow(face_images[i])
    plt.show()
