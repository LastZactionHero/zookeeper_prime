# Creates an average zero image from several known zeros
import constants
from os import listdir
from scipy import misc
import numpy

def remove_zero(image):
    return image - generate_zero_average_image()

def generate_zero_average_image():
    filenames = listdir(constants.IMAGE_64_PATH)

    X = []
    for filename in filenames:
        if(filename[0] == '.'):
            continue;

        image = misc.imread(constants.IMAGE_64_PATH + '/' + filename, mode='L')
        X.append(image)

    X = (numpy.array(X) / 256.0)

    average = sum(X) / len(filenames)
    return average
