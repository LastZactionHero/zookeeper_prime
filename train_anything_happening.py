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
from tflearn.layers.conv import conv_1d, max_pool_1d
from skimage.filters import roberts, sobel, scharr, prewitt

from tflearn.layers.estimator import regression

from scipy import misc
import matplotlib.pyplot as plt
import code

zero_image_path = '/Users/zach/Dropbox/machine_learning/image_trainer/known_zero_64'
image_path = '/Users/zach/Dropbox/machine_learning/image_trainer/training_images_64'
csv_filename = '/Users/zach/Dropbox/machine_learning/image_trainer/anything_happening.csv'

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

    rowY = [0,0]
    if(row[5] == '1'):
        rowY[0] = 1
    else:
        rowY[1] = 1
    Y.append(rowY)

# Remove events where nothing is happening to avoid local minima
removal_idx = []
null_count = 0
for idx, y in enumerate(Y):
    if(y[1] == 1):
        null_count += 1
        if(null_count % 4 <= 1):
            removal_idx.append(idx)

removal_idx.reverse()
for idx in removal_idx:
    del Y[idx]
    del filenames[idx]


# Read the images
X = []
for filename in filenames:
    image = misc.imread(image_path + '/' + filename)
    edge_image = image
    X.append(edge_image)

X = (numpy.array(X) / 256.0)

zero_image = misc.imread(zero_image_path + '/' + 'zero_daytime.jpg')
zero_image = numpy.array(zero_image) / 256.0
X = X - zero_image

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

# Since the image position remains consistent and are fairly similar, this can be spatially aware.
# Using a fully connected network directly, no need for convolution.
network = fully_connected(network, 2048, activation='relu')
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.00003)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X_train, y_train, n_epoch=100, shuffle=True, validation_set=(X_test, y_test),
          show_metric=True, batch_size=100, run_id='anything_happening_cnn')
model.save('model_anything_happening.tflearn')

code.interact(local=locals())