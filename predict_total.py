import constants
import sys
from scipy import misc
import numpy
from zero_average import remove_zero
import cPickle
from sklearn import linear_model
from network_anything_happening import build_model_anything_happening
from network_specific import build_model_specific
from tensorflow.python.framework import ops

# Read the image
filename = sys.argv[1]
X = [misc.imread(filename, mode='L')]
brightness = numpy.array(X).mean()

with open(constants.PICKLE_NIGHT_DAY, 'rb') as fid:
    clf = cPickle.load(fid)

is_nighttime = clf.predict(brightness)[0] > 0.5
if is_nighttime:
    print "> Too dark to see."
    exit()


X = (numpy.array(X) / 256.0)
X = remove_zero(X)

model_anything_happening = build_model_anything_happening()
model_anything_happening.load(constants.TFLEARN_ANYTHING_HAPPENING_FILENAME)

is_anything_happening = (numpy.argmax(model_anything_happening.predict(X)[0]) == 0)
if is_anything_happening == False:
    print "> Nothing is happening."
    exit()


ops.reset_default_graph()

model_specific = build_model_specific()
model_specific.load(constants.TFLEARN_SPECIFIC_FILENAME)

specific = numpy.argmax(model_specific.predict(X)[0])

animals = {
    0 : "Strcat",
    1 : "Malloc",
    2 : "Cody",
    3 : "Unknown"
}
the_animal = animals[specific]

print "> It's a %s!" % the_animal
