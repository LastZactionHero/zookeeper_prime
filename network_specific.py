import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.estimator import regression


def build_model_specific():
    ### IS ANY OF THIS NECESSARY FOR LIGHT/DARK? IN GENERAL W/ STAIONARY CAMERA?
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()

    # Specify shape of the data, image prep
    network = input_data(shape=[None, 52, 64],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

    # conv_2d incoming, nb_filter, filter_size
    # incoming: Tensor. Incoming 4-D Tensor.
    # nb_filter: int. The number of convolutional filters. # WHAT IS THIS?
    # filter_size: 'intor list ofints`. Size of filters.   # WHAT IS THIS?
    network = conv_1d(network, 512, 3, activation='relu')

    # (incoming, kernel_size)
    # incoming: Tensor. Incoming 4-D Layer.
    # kernel_size: 'intor list ofints`. Pooling kernel size.
    network = max_pool_1d(network, 2)

    network = conv_1d(network, 64, 3, activation='relu')
    network = conv_1d(network, 64, 3, activation='relu')
    network = max_pool_1d(network, 2)

    network = fully_connected(network, 512, activation='relu')

    network = dropout(network, 0.5)

    network = fully_connected(network, 4, activation='softmax')

    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.0003)

    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model
