# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:04:19 2018

@author: khanhle
"""



# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import confusion_matrix

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print(__doc__)

import h5py
import os
import sys
from keras.models import model_from_json

#define params
trn_file = sys.argv[1]
tst_file = sys.argv[2]
json_file = sys.argv[3]
h5_file = sys.argv[4]

nb_classes = 2
nb_kernels = 3
nb_pools = 2
window_sizes = 19

# load training dataset
dataset = np.loadtxt(trn_file, delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:window_sizes*20].reshape(len(dataset),1,20,window_sizes)
Y = dataset[:,window_sizes*20]

Y = np_utils.to_categorical(Y,nb_classes)
#print X,Y
#nb_classes = Y.shape[1]
#print nb_classes

# load testing dataset
dataset1 = np.loadtxt(tst_file, delimiter=",")
# split into input (X) and output (Y) variables
X1 = dataset1[:,0:window_sizes*20].reshape(len(dataset1),1,20,window_sizes)
Y1 = dataset1[:,window_sizes*20]
true_labels = np.asarray(Y1)

Y1 = np_utils.to_categorical(Y1,nb_classes)
#print('label : ', Y[i,:])

def cnn_model():
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape = (1,20,window_sizes)))
    model.add(Convolution2D(32, nb_kernels, nb_kernels))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(32, nb_kernels, nb_kernels, activation='relu'))
    # model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, nb_kernels, nb_kernels, activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, nb_kernels, nb_kernels, activation='relu'))
    # model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    ## add the model on top of the convolutional base
    #model.add(top_model)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(nb_classes))
    #model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # f = open('model_summary.txt','w')
    # f.write(str(model.summary()))
    # f.close()

    #model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

#plot_filters(model.layers[0],32,1)
# Fit the model
# save best weights
model = cnn_model()
#plot_model(model, to_file='model.png')

filepath = "weights.best.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
# balance data
model.fit(X, Y, nb_epoch=150, batch_size=10, class_weight = 'auto', validation_data=(X1,Y1), callbacks=[checkpointer])
## evaluate the model
scores = model.evaluate(X1, Y1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.load_weights(filepath)
predictions = model.predict_classes(X1)

print(confusion_matrix(true_labels, predictions))

#serialize model to JSON
model_json = model.to_json()
with open(json_file, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(h5_file)
print("Saved model to disk")

