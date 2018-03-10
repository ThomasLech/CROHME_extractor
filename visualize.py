import sys
import os
import argparse

import pickle

import math
import random
# Image processing
from skimage.feature import hog
# Data visualization
import matplotlib.pyplot as plt
# One-hot encoder/decoder
import one_hot

'constants'
outputs_rel_path = 'outputs'
train_dir = os.path.join(outputs_rel_path, 'train')
test_dir = os.path.join(outputs_rel_path, 'test')

ap = argparse.ArgumentParser()
ap.add_argument('-b', '--box_size', required=True, help="Specify a length of square box side.")
ap.add_argument('-n', '--n_samples', required=True, help="Specify the nubmer of samples to show.")
ap.add_argument('-c', '--columns', required=True, help="Specify the nubmer of columns.")
args = vars(ap.parse_args())

# Load pickled data
with open(os.path.join(train_dir, 'train.pickle'), 'rb') as train:
    print('Restoring training set ...')
    train_set = pickle.load(train)

with open(os.path.join(test_dir, 'test.pickle'), 'rb') as test:
    print('Restoring test set ...')
    test_set = pickle.load(test)

# Extract command-line arguments
box_size = int(args.get('box_size'))
n_samples = int(args.get('n_samples'))
n_cols = int(args.get('columns'))

# Load classes
classes = open('classes.txt', 'r').read().split()

'Compute number of rows with respect to number of both columns and samples provided by user'
rows_numb = math.ceil(n_samples / n_cols)

'Instanciate a figure to plot samples on'
figure, axis_arr = plt.subplots(rows_numb, n_cols, figsize=(12, 4))
figure.patch.set_facecolor((0.91, 0.91, 0.91))

sample_id = 0
for row in range(rows_numb):
    for col in range(n_cols):

        if sample_id < n_samples:
            'Generate random sample id'
            random_id = random.randint(0, len(test_set))
            training_sample = test_set[random_id]
            # Decode from one-hot format to string
            label = one_hot.decode(training_sample['label'], classes)

            axis_arr[row, col].imshow(training_sample['features'].reshape((box_size, box_size)), cmap='gray')
            axis_arr[row, col].set_title('Class: \"' + label + '\"', size=13, y=1.2)

        'Remove explicit axises'
        axis_arr[row, col].axis('off')

        sample_id += 1

'Adjust spacing between subplots and window border'
figure.subplots_adjust(hspace=1.4, wspace=0.2)
plt.savefig('visualization.png')

# Brings foreground
plt.show()
