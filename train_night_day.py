from network_night_day import build_model_night_day
import csv
import numpy
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import shuffle

from os import listdir
from scipy import misc

image_path = '/Users/zach/Dropbox/machine_learning/image_trainer/training_images_32'
csv_filename = '/Users/zach/Dropbox/machine_learning/image_trainer/night_day.csv'


# Read the images
filenames = listdir(image_path)

X = []
for filename in filenames:
    image = misc.imread(image_path + '/' + filename, mode='L')
    X.append(image)

X = (numpy.array(X) / 256.0)

# code.interact(local=locals())

# Read the csv
f = open(csv_filename, 'rb')
reader = csv.reader(f)

Y = []
for idx, row in enumerate(reader):
    # Skip the header row
    if(idx == 0):
        continue

    rowY = [0,0]
    if(row[1] == '1'):
        rowY[0] = 1
    elif(row[2] == '1'):
        rowY[1] = 1
    Y.append(rowY)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train, y_train = shuffle(X_train, y_train)


model = build_model_night_day()
model.fit(X_train, y_train, n_epoch=20, shuffle=True, validation_set=(X_test, y_test),
          show_metric=True, batch_size=100, run_id='night_day_cnn')
model.save('model_night_day.tflearn')
