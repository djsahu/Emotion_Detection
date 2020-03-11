# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:05:12 2020

@author: KIIT
"""

import urllib
import cv2
import numpy as np

URL = "http://100.77.240.100:8080/shot.jpg"

face_data= r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\data\haarcascades\haarcascade_frontalface_default.xml"

pathOut=r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\manual_emotion_data\train\sad"

face_images=[]

classifier = cv2.CascadeClassifier(face_data)

ret=True
c=0
while(ret):
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),dtype=np.uint8)
    
    frame = cv2.imdecode(image,-1)
    faces = classifier.detectMultiScale(frame,2,2)
    for x,y,w,h in faces:
        if(c<700):
            face_img= frame[y:y+h,x:x+w]
            face_images.append(face_img)
            cv2.imwrite(pathOut + "\\dev_sad%d.jpg" % c, face_img)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
            c=c+1
        
    cv2.imshow('video',frame)
    if cv2.waitKey(24)==ord('q'):
        break

cv2.destroyAllWindows()