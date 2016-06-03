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

zero_image_path = '/Users/zach/Dropbox/machine_learning/image_trainer/known_zero_64'
zero_image = misc.imread(zero_image_path + '/' + 'zero_daytime.jpg')
zero_image = numpy.array(zero_image) / 256.0

X = X - zero_image

### IS ANY OF THIS NECESSARY FOR LIGHT/DARK? IN GENERAL W/ STAIONARY CAMERA?
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()

# Specify shape of the data, image prep
network = input_data(shape=[None, 52, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Since the image position remains consistent and are fairly similar, this can be spatially aware.
# Using a fully connected network directly, no need for convolution.
network = fully_connected(network, 2048, activation='relu')
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.00003)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('model_anything_happening.tflearn')

print model.predict(X)