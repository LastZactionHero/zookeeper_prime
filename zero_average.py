# Creates an average zero image from several known zeros
from os import listdir
from scipy import misc
import numpy

def remove_zero(image):
    return image - generate_zero_average_image()

def generate_zero_average_image():
    zero_image_path = '/Users/zach/Dropbox/machine_learning/image_trainer/known_zero_64'
    filenames = listdir(zero_image_path)

    X = []
    for filename in filenames:
        if(filename[0] == '.'):
            continue;

        image = misc.imread(zero_image_path + '/' + filename, mode='L')
        X.append(image)

    X = (numpy.array(X) / 256.0)

    average = sum(X) / len(filenames)
    return average
