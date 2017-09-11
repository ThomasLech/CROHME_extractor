import sys
import os

import pickle

import math
import random
# Image processing
from skimage.feature import hog
# Data visualization
import matplotlib.pyplot as plt

'constants'
outputs_rel_path = 'outputs'
train_dir = os.path.join(outputs_rel_path, 'train')
test_dir = os.path.join(outputs_rel_path, 'test')
validation_dir = os.path.join(outputs_rel_path, 'validation')

if __name__ == '__main__':

    'parse cmd input'
    print(' # Script flags:', '<number_of_samples>', '<number_of_columns=4>', '\n')

    'parse number_of_samples argument'
    if len(sys.argv) < 2:
        print('\n ! Usage:', sys.argv[0], '<number_of_samples>', '<number_of_columns=4>', '\n')
        exit()

    try:
        number_of_samples = int(sys.argv[1])
    except Exception as e:
        print(e)
        exit()

    'parse cols_numb argument'
    cols_numb = 4
    if len(sys.argv) == 3:

        try:
            cols_numb = int(sys.argv[2])
        except Exception as e:
            print(e)
            exit()

    'Load pickled data'
    with open(os.path.join(train_dir, 'train.pickle'), 'rb') as train:
        print('Restoring training set ...')
        train_set = pickle.load(train)

    with open(os.path.join(test_dir, 'test.pickle'), 'rb') as test:
        print('Restoring test set ...')
        test_set = pickle.load(test)

    with open(os.path.join(validation_dir, 'validation.pickle'), 'rb') as validation:
        print('Restoring validation set ...')
        validation_set = pickle.load(validation)
    # Get size of the original box that was flattened
    box_size = int(math.sqrt(train_set[0]['features'].size))

    'Compute number of rows with respect to number of both columns and samples provided by user'
    rows_numb = math.ceil(number_of_samples / cols_numb)

    'Instanciate a figure to plot samples on'
    figure, axis_arr = plt.subplots(rows_numb, cols_numb, figsize=(12, 4))
    figure.patch.set_facecolor((0.91, 0.91, 0.91))

    sample_id = 0
    for row in range(rows_numb):
        for col in range(cols_numb):

            if sample_id < number_of_samples:

                'Generate random sample id'
                random_id = random.randint(0, len(train_set))

                axis_arr[row, col].imshow(train_set[random_id]['features'].reshape((box_size, box_size)), cmap='gray')
                axis_arr[row, col].set_title('Class: \"' + train_set[random_id]['label'] + '\"', size=13, y=1.2)

            'Remove explicit axises'
            axis_arr[row, col].axis('off')

            sample_id += 1

    'Adjust spacing between subplots and window border'
    figure.subplots_adjust(hspace=1.4, wspace=0.2)

    # Brings foreground
    plt.show()
