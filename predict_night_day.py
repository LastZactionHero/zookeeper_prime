import code

import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.cross_validation import train_test_split
import numpy
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import sys
from scipy import misc

# Read the image
filename = sys.argv[1]
X = [misc.imread(filename)]
X = (numpy.array(X) / 256.0)

### IS ANY OF THIS NECESSARY FOR LIGHT/DARK? IN GENERAL W/ STAIONARY CAMERA?
img_prep = ImagePreprocessing()
# img_prep.add_featurewise_zero_center()
# img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
# img_aug.add_random_flip_leftright()
 
# Specify shape of the data, image prep
network = input_data(shape=[None, 288, 352, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# conv_2d incoming, nb_filter, filter_size
# incoming: Tensor. Incoming 4-D Tensor.
# nb_filter: int. The number of convolutional filters. # WHAT IS THIS?
# filter_size: 'intor list ofints`. Size of filters.   # WHAT IS THIS?
network = conv_2d(network, 32, 3, activation='relu')

# (incoming, kernel_size)
# incoming: Tensor. Incoming 4-D Layer.
# kernel_size: 'intor list ofints`. Pooling kernel size.
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 512, activation='relu')

network = dropout(network, 0.5)

network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.00003)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('model_night_day.tflearn')

print model.predict(X)