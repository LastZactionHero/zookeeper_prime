import numpy
import sys
from scipy import misc
import cPickle
from sklearn import linear_model
import numpy as np

# Read the image
filename = sys.argv[1]
image = [misc.imread(filename, mode='L')]
brightness = np.array(image).mean()

with open('model_night_day.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

print clf.predict(brightness)[0]
