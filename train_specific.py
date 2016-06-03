import tensorflow as tf
import csv
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.cross_validation import train_test_split
import numpy
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.estimator import regression

from scipy import misc

import code

image_path = '/Users/zach/Dropbox/machine_learning/image_trainer/training_images_64'
csv_filename = '/Users/zach/Dropbox/machine_learning/image_trainer/specific.csv'

filenames = []

# Read the csv
f = open(csv_filename, 'rb')
reader = csv.reader(f)

Y = []
for idx, row in enumerate(reader):
    # Skip the header row
    if(idx == 0):
        continue

    filenames.append(row[0])
    rowY = [0,0,0,0]
    if(row[1] == 'Y'):
        rowY[0] = 1
    if(row[2] == 'Y'):
        rowY[1] = 1
    if(row[3] == 'Y'):
        rowY[2] = 1
    if(row[4] == 'Y'):
        rowY[3] = 1
    Y.append(rowY)


# Read the images
X = []
for filename in filenames:
    image = misc.imread(image_path + '/' + filename)
    X.append(image)

X = (numpy.array(X) / 256.0)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train, y_train = shuffle(X_train, y_train)


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

# conv_2d incoming, nb_filter, filter_size
# incoming: Tensor. Incoming 4-D Tensor.
# nb_filter: int. The number of convolutional filters. # WHAT IS THIS?
# filter_size: 'intor list ofints`. Size of filters.   # WHAT IS THIS?
network = conv_2d(network, 256, 3, activation='relu')

# (incoming, kernel_size)
# incoming: Tensor. Incoming 4-D Layer.
# kernel_size: 'intor list ofints`. Pooling kernel size.
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 512, activation='relu')

network = dropout(network, 0.5)

network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X_train, y_train, n_epoch=50, shuffle=True, validation_set=(X_test, y_test),
          show_metric=True, batch_size=25, run_id='specific_cnn')
model.save('model_specific.tflearn')