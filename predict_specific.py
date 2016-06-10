import constants
import numpy
import sys
from scipy import misc
from network_specific import build_model_specific
from zero_average import remove_zero

# Read the image
filename = sys.argv[1]
X = [misc.imread(filename, mode='L')]
X = (numpy.array(X) / 256.0)

X = remove_zero(X)

model = build_model_specific()
model.load(constants.TFLEARN_SPECIFIC_FILENAME)

print model.predict(X)
