import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def build_model_night_day():
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
    return model
