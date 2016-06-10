# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import constants
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import misc
from sklearn import linear_model
import cPickle

filenames = []

# Read the csv
f = open(constants.NIGHT_DAY_CSV_FILENAME, 'rb')
reader = csv.reader(f)

Y = []
for idx, row in enumerate(reader):
    # Skip the header row
    if(idx == 0):
        continue
    filenames.append(row[0])

    y = 0
    if(row[1] == '1'):
        y = 1

    Y.append(y)
Y = np.array(Y)

X = np.array([])
for filename in filenames:
    image = misc.imread(constants.IMAGE_64_PATH + '/' + filename, mode='L')
    avg_brightness = np.matrix(image).mean()
    X = np.append(X, avg_brightness)
X = np.array([X]).transpose()

clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, Y)

# Save the training
with open(constants.PICKLE_NIGHT_DAY, 'wb') as fid:
    cPickle.dump(clf, fid)

# Plot all data
plt.scatter(X, Y, c='aqua', label='Actual')
plt.hold('on')

# Plot test predictions
X_range = range(int(min(X)),int(max(X)))
X_range = np.array(map(lambda x: [x], X_range))
plt.plot(X_range, clf.predict(X_range), c='red', label='Prediction', linewidth=1)

plt.xlabel('Brightness')
plt.ylabel('Nighttime')

plt.legend()
plt.show()
