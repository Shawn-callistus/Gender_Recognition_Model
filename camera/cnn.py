import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras import models,layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from sklearn.model_selection import train_test_split

# In[16]:


x=pickle.load(open("x.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))

X,xtest,Y,ytest=train_test_split(x,y,test_size=0.3,random_state=69)

# In[17]:


x=x/255
xtest=xtest/255


# In[21]:


cnn=models.Sequential([
    layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu",input_shape=(32,32,1)),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"), 
    layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),   
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(100,activation="relu"),
    layers.Dense(2,activation="sigmoid")
])

print(cnn.summary())

# In[24]:


cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[26]:


cnn.fit(x,y=np.array(y),batch_size=64,epochs=10,validation_split=0.1)


# In[27]:


pred=cnn.predict(xtest)


# In[28]:


print(pred[0])


# In[29]:


c=1
for i in range(10,20):
    figt=plt.figure(figsize=(3,3))
    plt.subplot(2,5,c)
    plt.imshow(xtest[i])
    plt.xlabel(np.argmax(pred[i]))
    c+=1


# In[30]:

cnn.save("classification.model")

# In[ ]:




