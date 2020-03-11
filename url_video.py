# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:43:24 2020

@author: KIIT
"""

import urllib
import cv2
import numpy as np

URL = "http://192.168.42.129:8080/shot.jpg"

ret = True
while ret:
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),dtype=np.uint8)
    frame = cv2.imdecode(image,-1)
    
    cv2.imshow('video',frame)
        
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()

