#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import packages
import os
import pandas as pd
from sklearn.model_selection import KFold
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from numpy import asarray
import numpy as np


# In[ ]:


#make empty lists to prepare input and outputs data
path = []
model = []
airplane = []


# In[ ]:


#prepare input and outputs data (store picture's path and validation outputs)
def each_plane(zmodel, zairplane, zpath):
    corepath = "C:/Users/antoi/Google Drive/Computer Vision/airplane scanner/" + zmodel + "/" + zairplane + "/" + zpath + "/"
    arr = os.listdir(corepath)
    curr_path = []
    for i in range(len(arr)):
        curr_path.append(corepath + arr[i])
    path.extend(curr_path)
    
    curr_model = []
    for i in range(len(arr)):
        curr_model.append(zmodel)
    model.extend(curr_model)
    
    curr_airplane = []
    for i in range(len(arr)):
        curr_airplane.append(zairplane)
    airplane.extend(curr_airplane)


# In[ ]:


each_plane("civi", "727", "boeing 727 airplane")
each_plane("civi", "707", "boeing 707 airplane")
each_plane("civi", "380", "airbus 380 airplane")
each_plane("civi", "320", "airbus 320 airplane")

each_plane("mili", "su-37", "su-37 airplane")
each_plane("mili", "f-22", "f-22 airplane")
each_plane("mili", "mirage-2000", "mirage 2000 airplane")
each_plane("mili", "rafale", "rafale airplane")


# In[ ]:


#transform the pictures into a list of array
X = []

for i in range(len(path)):
    img = cv2.imread(path[i])
    img = asarray(img)
    mean = img.mean()
    img = img - mean
    X.append(img)
    
X = np.asarray(X).astype(np.float32)


# In[ ]:


#transform outputs label intop numbers; make a dictionary of relation between labels and their given number.
from sklearn import preprocessing

le2 = preprocessing.LabelEncoder()
le2.fit(airplane)

y = le2.transform(airplane)
y = np.asarray(y).astype(np.float32)

y_map = dict(zip(le2.classes_, le2.transform(le2.classes_)))
print(y_map)
y_keys, y_values = zip(*y_map.items())
print(y_keys)
print(y_values)


# In[ ]:


#define the number of fold for the kfold
kf = KFold(n_splits=5)
kf.get_n_splits(X)
print(kf)
KFold(n_splits=5, random_state=None, shuffle=False)


# In[ ]:


#actually fold the data
for train_index, test_index in kf.split(X):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[ ]:


#prepare inputs and outputs shape values
n_inputs = X.shape[1]
n_outputs = len(list(le2.classes_))
entry = (X.shape[1], X.shape[2], X.shape[3])
exit = len(list(le2.classes_))


# In[ ]:


#import model packages
from keras.layers import Dense
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#define the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=entry))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(exit, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
    return model

#train the model
def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], exit
    cv = kf
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        #call the previously defined model
        model = get_model(n_inputs, n_outputs)
        #train the model
        print('Train...')
        model.fit(X_train, y_train, verbose=1, epochs=10)
        #test the model, return a confusion matrix
        print('Evaluate...')
        y_pred = model.predict_classes(X_test, verbose=1)
        y_test = np.asarray(y_test).astype(int)
        zz = confusion_matrix(y_test, y_pred, labels=y_values)
        zz = pd.DataFrame(zz)
        zz = zz.set_axis(list(le2.classes_), axis='columns', inplace=False)
        zz = zz.set_axis(list(le2.classes_), axis='index', inplace=False)
        print(zz)
    return results


# In[ ]:


#compile and train the model
results = evaluate_model(X, y)


# In[ ]:


#[print(i.shape, i.dtype) for i in model.inputs]
#[print(o.shape, o.dtype) for o in model.outputs]
#[print(l.name, l.input_shape, l.dtype) for l in model.layers]

