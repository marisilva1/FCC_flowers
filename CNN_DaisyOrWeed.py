#!/usr/bin/env python
# coding: utf-8

# In[22]:


import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

# I restarted the kernel (oh no) and now none of my import keras things are working - google 
# tells me this is because I'm not all in the same virtual envrionment, but I installed with 
# pip so this shouldn't be the problem.

# ...Naturally, turning my laptop off and back on fixed the problem.


# In[23]:


#GET DATA

trainPath="/Users/marisilva/tf_files/flowers/"
valPath="/Users/marisilva/tf_files/val_flowers/"

# The ImageDataGenerator part of preprocessing will generate batches of tensor image data 
# with real-time data augmentation (cool). The data will be looped over (in batches).
training_data = keras.preprocessing.image.ImageDataGenerator()
validation_data = keras.preprocessing.image.ImageDataGenerator()

# Using flow_from _directory YIELDS tuples of (x, y), where x is a numpy array containing a 
# batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of 
# corresponding labels.
tData = training_data.flow_from_directory(trainPath)
vData = validation_data.flow_from_directory(valPath)


# In[28]:


#Start to build model
model = Sequential()

#Convolutional layer is always first. Reads an image similar to a flashlight, whch shines over a "receptive field."
#Filter is an array of weights (as we learned) which is matrix-multiplied with the pixels as the flashlight "shines" over all areas of the image
#Then we're left with a feature map.
model.add(Conv2D(64,(3,3), input_shape=(256, 256,3), padding='same'))

#Next add a relu activation layer which will replace all negative values in the feature map with ZERO.
model.add(Activation('relu'))

#Now there's a max pooling layer - reducing the dimension of the feature maps(s) and just downsizing the most important bits
model.add(MaxPooling2D(pool_size=(2,2)))

#Now REPEAT so more fine details can be detected
model.add(Conv2D(32,(3,3), input_shape=(256, 256,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Repeat again
model.add(Conv2D(64,(3,3), input_shape=(256, 256,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout layer to prevent overfitting. First flatten map to one dimension.
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

#Now apply a sigmoid function, so we are able to convert to PROBABILITIES
model.add(Activation('sigmoid'))


# In[29]:


#Now dealing with loss.
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'],)


# In[31]:


#TRAIN MODEL!
import image

# The fit generator trains the model on data generated batch-by-batch by a Python generator 
# (or an instance of Sequence).
model.fit_generator(
        tData,
        steps_per_epoch=64,
        epochs=30,
        verbose=1)

#save weights, because we can reference them in the future 
model.save_weights('models/CNNplants.h5')


# In[11]:


# Finally, test with predictions. This is where my code will interact with the user, 
# theoretically, although in this instance the user needs to set the whole thing up...

print("Include the path to the flower image you'd like to identify:")
print("(must be a .jpg, and you don't need to put quotes around the path!)")
imageName = input()

oneImage = keras.preprocessing.image.ImageDataGenerator()
image = oneImage.flow_from_directory(imageName)
prediction = model.predict(image)
print(prediction)

print("If your prediction reads anything less than .75 corresponding to 'daisy', feel free to spray that sucker with herbicide.")


# In[ ]:




