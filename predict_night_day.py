from network_night_day import build_model_night_day
import numpy
import sys
from scipy import misc

# Read the image
filename = sys.argv[1]
X = [misc.imread(filename, mode='L')]
X = (numpy.array(X) / 256.0)

model = build_model_night_day()
model.load('model_night_day.tflearn')

print model.predict(X)
