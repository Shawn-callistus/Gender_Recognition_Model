#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import pickle


# In[3]:
#model=pickle.load(open("model.pickle","rb"))
model=load_model("classification.model")



# In[4]:


classes=["male","female"]


# In[5]:

face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera=cv2.VideoCapture(0)
while True:
    rec,frame=camera.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.1,3)
    for (x,y,w,h) in faces:
        endx=x+w
        endy=y+h
        cv2.rectangle(frame,(x,y),(endx,endy),(255,0,0),2)
        face_crop=np.copy(frame[y:endy,x:endx])
        face_crop=cv2.cvtColor(face_crop,cv2.COLOR_BGR2GRAY)
        face_crop=cv2.resize(face_crop,(32,32))
        face_crop=face_crop.astype("float")/255.0
        face_crop=img_to_array(face_crop)
        face_crop=np.expand_dims(face_crop,axis=0)
        print(face_crop.shape)
        conf=model.predict(face_crop)
        idx=np.argmax(conf)
        label=classes[idx]
        Y=y-10 if y-10>10 else y+10
        cv2.putText(frame,label,(x,Y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        
        
        
    cv2.imshow("out",frame)
    if cv2.waitKey(1)==ord("q"):
        break


# In[ ]:





# In[ ]:




