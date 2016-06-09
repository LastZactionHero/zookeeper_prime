from network_anything_happening import build_model_anything_happening
from network_night_day import build_model_night_day
from network_specific import build_model_specific
import sys
import numpy
from scipy import misc
from zero_average import remove_zero

# Read the image
filename = sys.argv[1]
X = [misc.imread(filename, mode='L')]
X = (numpy.array(X) / 256.0)
X_no_zero = remove_zero(X)

model_night_day = build_model_night_day()
model_night_day.load('model_night_day.tflearn')

prediction_night_day = model_night_day.predict(X)[0]
print(prediction_night_day)
nighttime = prediction_night_day.index(max(prediction_night_day)) == 0
print("Is nighttime: %d" % nighttime)
if(nighttime):
    exit()

model_anything_happening = build_model_anything_happening()
model_anything_happening.load('model_anything_happening.tflearn')
predict_anything_happening = model_anything_happening.predict(X_no_zero)[0]
print(predict_anything_happening)
anything_happening = predict_anything_happening.index(max(predict_anything_happening)) == 0
print("Is anything happening: %d" % anything_happening)
