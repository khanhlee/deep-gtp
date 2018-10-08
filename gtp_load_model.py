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
from keras.models import model_from_json

#define params
tst_file = sys.argv[1]
json_model = sys.argv[2]
h5_model = sys.argv[3]

nb_classes = 2
nb_kernels = 3
nb_pools = 2
window_sizes = 19

# load training dataset
# dataset = numpy.loadtxt(trn_file, delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:window_sizes*20].reshape(len(dataset),1,20,window_sizes)
# Y = dataset[:,window_sizes*20]

# Y = np_utils.to_categorical(Y,nb_classes)
#print X,Y
#nb_classes = Y.shape[1]
#print nb_classes

# load testing dataset
dataset1 = numpy.loadtxt(tst_file, delimiter=",")
# split into input (X) and output (Y) variables
X1 = dataset1[:,0:window_sizes*20].reshape(len(dataset1),1,20,window_sizes)
Y1 = dataset1[:,window_sizes*20]

Y1 = np_utils.to_categorical(Y1,nb_classes)

def cnn_model():
    model = Sequential()

    model.add(Dropout(0.2, input_shape = (1,20,window_sizes)))
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
    #model.add(Dropout(0.5))
    model.add(Dense(128))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    #model.add(BatchNormalization())
    model.add(Activation('softmax'))

    #model.summary()

    #model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def mlp_model():
    model = Sequential() # The Keras Sequential model is a linear stack of layers.
    model.add(Dense(100, init='uniform', input_dim=400)) # Dense layer
    model.add(Activation('tanh')) # Activation layer
    model.add(Dropout(0.5)) # Dropout layer
    model.add(Dense(100, init='uniform')) # Another dense layer
    model.add(Activation('tanh')) # Another activation layer
    model.add(Dropout(0.5)) # Another dropout layer
    model.add(Dense(2, init='uniform')) # Last dense layer
    model.add(Activation('softmax')) # Softmax activation at the end
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # Using Nesterov momentum
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) # Using logloss
    return model	

# define 10-fold cross validation test harness
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# cvscores = []
# for train, test in kfold.split(X, Y):
#     model = cnn_model()
#     # save best weights
#     filepath = "weights.best.hdf5"
#     checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
#     # balance data
#     class_weight = {0:1.,1:5.}
    
#     ## evaluate the model
#     model.fit(X[train], np_utils.to_categorical(Y[train],nb_classes), nb_epoch=50, batch_size=10, class_weight = 'auto', callbacks=[checkpointer], verbose=0)
#     # evaluate the model
#     scores = model.evaluate(X[test], np_utils.to_categorical(Y[test],nb_classes), verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)
#     #prediction
#     #model.load_weights(filepath)
#     predictions = model.predict(X[test])
#     output = np_utils.categorical_probas_to_classes(predictions)
#     pred_labels = numpy.asarray(output)
#     true_labels = numpy.asarray(Y[test])
# Fit the model
# save best weights
# model = cnn_model()
# filepath = "weights.best.hdf5"
# checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)
# # balance data
# model.fit(X, Y, nb_epoch=200, batch_size=10, class_weight = 'auto', validation_data=(X1,Y1), callbacks=[checkpointer])
# ## evaluate the model
# scores = model.evaluate(X1, Y1)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# model.load_weights(filepath)

# # serialize model to JSON
# model_json = model.to_json()
# with open("final_model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("final_model.h5")
# print("Saved model to disk")

# load json and create model
json_file = open(json_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(h5_model)
print("Loaded model from disk")
predictions = loaded_model.predict(X1)

print(predictions)
f = open('output.csv','w')
output = np_utils.categorical_probas_to_classes(predictions)
for x in output:
    f.write(str(x) + '\n')
f.close()
