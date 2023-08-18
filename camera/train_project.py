#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# In[21]:
xdata=[]
ydata=[]
face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
datadir="C:/Users/monik/OneDrive/Desktop/project/Train"
labels=["male","female"]
train=[]
for label in labels:
    path=os.path.join(datadir,label)
    labeln=labels.index(label)
    for img in os.listdir(path):
        imgarr=cv2.imread(os.path.join(path,img))
        faces=face.detectMultiScale(imgarr)
        for (x,y,w,h) in faces:
            endx=x+w
            endy=y+h
            face_crop=np.copy(imgarr[y:endy,x:endx])
            face_crop=cv2.cvtColor(face_crop,cv2.COLOR_BGR2GRAY)
            face_crop=cv2.resize(face_crop,(32,32))
            train.append([face_crop,labeln])
# In[22]:


import random
random.shuffle(train)


# In[23]:
#figt=plt.figure(figsize=(3,3))
#plt.subplot(2,5,1)
#plt.imshow(train[0][0])
#print(train[0][1])


# In[32]:


x=[]
y=[]
for i,j in train:
        x.append(i)
        y.append(j)




# In[43]:


x=np.reshape(x,(-1,32,32,1))


# In[46]:


import pickle 
pickle_out=open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()
pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()



