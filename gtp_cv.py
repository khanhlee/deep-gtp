import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import h5py
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#define params
trn_file = sys.argv[1]
tst_file = sys.argv[2]

nb_classes = 2
nb_kernels = 3
nb_pools = 2
window_sizes = 19

# load training dataset
dataset = numpy.loadtxt(trn_file, delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:window_sizes*20].reshape(len(dataset),1,20,window_sizes)
Y = dataset[:,window_sizes*20]

Y = np_utils.to_categorical(Y,nb_classes)
#print X,Y
#nb_classes = Y.shape[1]
#print nb_classes

# load testing dataset
dataset1 = numpy.loadtxt(tst_file, delimiter=",")
# split into input (X) and output (Y) variables
X1 = dataset1[:,0:window_sizes*20].reshape(len(dataset1),1,20,window_sizes)
Y1 = dataset1[:,window_sizes*20]
true_labels = numpy.asarray(Y1)

Y1 = np_utils.to_categorical(Y1,nb_classes)

def cnn_model():
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape = (1,20,window_sizes)))
    model.add(Convolution2D(32, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, nb_kernels, nb_kernels, activation='relu'))
    model.add(MaxPooling2D(strides=(nb_pools, nb_pools), dim_ordering="th"))

    ## add the model on top of the convolutional base
    #model.add(top_model)
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(128))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

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

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    model = cnn_model()   
    ## evaluate the model
    model.fit(X[train], np_utils.to_categorical(Y[train],nb_classes), nb_epoch=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], np_utils.to_categorical(Y[test],nb_classes), verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    #prediction
    #model.load_weights(filepath)
    true_labels = numpy.asarray(Y[test])
    predictions = model.predict_classes(X[test])
    print(confusion_matrix(true_labels, predictions))

