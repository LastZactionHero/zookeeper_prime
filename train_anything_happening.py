from zero_average import remove_zero
import matplotlib.pyplot as plt
from scipy import misc
import csv
import numpy
from sklearn.cross_validation import train_test_split
from tflearn.data_utils import shuffle
from network_anything_happening import build_model_anything_happening
from print_results import print_results


image_path = '/Users/zach/Dropbox/machine_learning/image_trainer/training_images_64'
csv_filename = '/Users/zach/Dropbox/machine_learning/image_trainer/anything_happening.csv'

filenames = []

# Read the csv
f = open(csv_filename, 'rb')
reader = csv.reader(f)

Y = []
for idx, row in enumerate(reader):
    # Skip the header row
    if(idx == 0):
        continue

    filenames.append(row[0])

    rowY = [0,0]
    if(row[5] == '1'):
        rowY[0] = 1
    else:
        rowY[1] = 1
    Y.append(rowY)

# Remove events where nothing is happening to avoid local minima
removal_idx = []
null_count = 0
for idx, y in enumerate(Y):
    if(y[1] == 1):
        null_count += 1
        if(null_count % 4 <= 1):
            removal_idx.append(idx)

removal_idx.reverse()
for idx in removal_idx:
    del Y[idx]
    del filenames[idx]


# Read the images
X = []
for filename in filenames:
    image = misc.imread(image_path + '/' + filename, 'L')
    edge_image = image
    X.append(edge_image)

X = (numpy.array(X) / 256.0)

X = remove_zero(X)


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train, y_train = shuffle(X_train, y_train)

model = build_model_anything_happening()
model.fit(X_train, y_train, n_epoch=100, shuffle=True, validation_set=(X_test, y_test),
          show_metric=True, batch_size=100, run_id='anything_happening_cnn')
model.save('model_anything_happening.tflearn')

print_results(X, Y, model)
