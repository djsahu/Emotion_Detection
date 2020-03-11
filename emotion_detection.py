# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:42:27 2020

@author: KIIT
"""

#import urllib
import cv2
from keras.models import load_model

classifier = cv2.CascadeClassifier(r"D:\Documents\Complete-Python-3-Bootcamp-master\Datasets\data\haarcascades\haarcascade_frontalface_default.xml")
model = load_model(r"D:\Documents\My Projects\Emotion_Detection\emotion_model_95.h5")
URL = "http://100.79.92.239:8080/shot.jpg"


import pickle
with open(r"D:\Documents\My Projects\Emotion_Detection\train_labels_wo_tl_tr.p", 'rb') as f:
  train_labels = pickle.load(f)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train_labels = le.fit_transform(train_labels)



def preprocess(img):
    img = cv2.resize(img,(256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img.reshape(1,256,256,1)
    img = img/255
    return img

#ret = True
#while ret:
#    img = urllib.request.urlopen(URL)
#    image = np.array(bytearray(img.read()),dtype=np.uint8)
#    frame = cv2.imdecode(image,-1)
#
#    faces = classifier.detectMultiScale(frame,1.3,5)
#    
#    for x,y,w,h in faces:
#        face = frame[y:y+h+10,x:x+h+10]
#        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
#        cv2.putText(frame,le.inverse_transform(model.predict_classes(preprocess(face))[0]),(x,y),
#                                         cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
#    cv2.imshow('video',frame)
#        
#    if cv2.waitKey(1)==ord('q'):
#        break
#    
#cv2.destroyAllWindows()

cap= cv2.VideoCapture(0)

ret=True

while(ret):
    if cap.isOpened():
        ret,frame=cap.read()
        faces = classifier.detectMultiScale(frame,2,2)
        
        for x,y,w,h in faces:
            face_img= frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
            cv2.putText(frame,le.inverse_transform(model.predict_classes(preprocess(face_img)))[0],(x,y),
                                         cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            
        cv2.imshow('video',frame)
        if cv2.waitKey(30)==ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
