# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:12:20 2020

@author: KIIT
"""

import urllib
import cv2
import numpy as np


face_data = r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\data\haarcascades\haarcascade_frontalface_default.xml"

URL = "http://192.168.42.129:8080/shot.jpg"

pathOut = r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\manual_emotion_data\All"

def preprocess(img):
    img = cv2.resize(img,(300,300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

classifier = cv2.CascadeClassifier(face_data)

data = []
ret = True
c=0

while ret:
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),dtype=np.uint8)
    frame = cv2.imdecode(image,-1)
    faces = classifier.detectMultiScale(frame,2,2)

    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        if len(data)<250:
            data.append(preprocess(face))
            cv2.imwrite(pathOut + "\\dj_son_sur%d.jpg" % c, preprocess(face))
            c=c+1
        else:
            cv2.putText(frame,'done',(100,100),
                        cv2.FONT_HERSHEY_PLAIN,4,
                        (255,255,255),5)
        
    cv2.imshow('video',frame)

    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()

