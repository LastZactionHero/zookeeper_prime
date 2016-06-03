import code

import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.cross_validation import train_test_split
import numpy
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
import sys
from scipy import misc

# Read the image
filename = sys.argv[1]
X = [misc.imread(filename, mode='L')]
X = (numpy.array(X) / 256.0)

### IS ANY OF THIS NECESSARY FOR LIGHT/DARK? IN GENERAL W/ STAIONARY CAMERA?
img_prep = ImagePreprocessing()
# img_prep.add_featurewise_zero_center()
# img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
# img_aug.add_random_flip_leftright()
 
# Specify shape of the data, image prep
network = input_data(shape=[None, 26, 32],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0003)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('model_night_day.tflearn')

print model.predict(X)