import constants
import tensorflow as tf
import csv
from sklearn.cross_validation import train_test_split
import numpy
from scipy import misc
from network_specific import build_model_specific
from print_results import print_results
from zero_average import remove_zero
from tflearn.data_utils import shuffle



filenames = []

# Read the csv
f = open(constants.SPECIFIC_CSV_FILENAME, 'rb')
reader = csv.reader(f)

Y = []
for idx, row in enumerate(reader):
    # Skip the header row
    if(idx == 0):
        continue

    filenames.append(row[0])
    rowY = [0,0,0,0]
    if(row[1] == 'Y'):
        rowY[0] = 1
    elif(row[2] == 'Y'):
        rowY[1] = 1
    elif(row[3] == 'Y'):
        rowY[2] = 1
    elif(row[4] == 'Y'):
        rowY[3] = 1
    Y.append(rowY)


# Read the images
X = []
for filename in filenames:
    image = misc.imread(constants.IMAGE_64_PATH + '/' + filename, mode='L')
    X.append(image)

X = (numpy.array(X) / 256.0)
X = remove_zero(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train, y_train = shuffle(X_train, y_train)

model = build_model_specific();
model.fit(X_train, y_train, n_epoch=100, shuffle=True, validation_set=(X_test, y_test),
          show_metric=True, batch_size=25, run_id='specific_cnn')
model.save(constants.TFLEARN_SPECIFIC_FILENAME)

print_results(X, Y, model)
