import numpy
import sys
from scipy import misc
from network_anything_happening import build_model_anything_happening
from zero_average import remove_zero

# Read the image
filename = sys.argv[1]
X = [misc.imread(filename, mode='L')]
X = (numpy.array(X) / 256.0)

X = remove_zero(X)

model = build_model_anything_happening()
model.load('model_anything_happening.tflearn')

print model.predict(X)
