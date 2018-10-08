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

# load testing dataset
dataset1 = numpy.loadtxt(tst_file, delimiter=",")
# split into input (X) and output (Y) variables
X1 = dataset1[:,0:window_sizes*20].reshape(len(dataset1),1,20,window_sizes)
Y1 = dataset1[:,window_sizes*20]

Y1 = np_utils.to_categorical(Y1,nb_classes)

# load json and create model
json_file = open(json_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(h5_model)
print("Loaded model from disk")
predictions = loaded_model.predict_classes(X1)

print(predictions)
f = open('output.csv','w')
for x in predictions:
    f.write(str(x) + '\n')
f.close()
