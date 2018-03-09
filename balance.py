'''
This script makes class_infos more balanced.
'''
import os
import pickle
import one_hot
from random import shuffle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

outputs_dir = 'outputs'
train_out_dir = os.path.join(outputs_dir, 'train')
test_out_dir = os.path.join(outputs_dir, 'test')
box_size = 50
# Balance ratio
b_ratio = 0.5
batch_size = 50

# Load data
with open(os.path.join(train_out_dir, 'train.pickle'), 'rb') as data:
    train = pickle.load(data)
with open(os.path.join(test_out_dir, 'test.pickle'), 'rb') as data:
    test = pickle.load(data)

print('Training set size:', len(train))
print('Testing set size:', len(test))

# Initialize keras image generator
datagen = ImageDataGenerator(rotation_range=5, shear_range=0.2)

# Load all class_infos that were extracted
classes = [label.strip() for label in list(open('classes.txt', 'r'))]
class_infos = [{'class': class_name, 'occurrences': 0} for class_name in classes]

for train_sample in train:
    label = one_hot.decode(train_sample['label'], classes)
    # Find index of this label in class_infos list
    class_idx = classes.index(label)
    # Update the number of occurrences
    class_infos[class_idx]['occurrences'] += 1

# Sort class_infos by occurrences
class_infos = sorted(class_infos, key=lambda class_info: class_info['occurrences'], reverse=True)
max_occurances = class_infos[0]['occurrences']
min_occurances = class_infos[len(class_infos)-1]['occurrences']
for class_info in class_infos:
    class_info['deviation'] = max_occurances - class_info['occurrences']

print('====================== Distribution of classes ======================')
for label in class_infos:
    print('CLASS: {}; occurrences: {}; deviation: {}'.format(label['class'], label['occurrences'], label['deviation']))
print('Max occurrences:', max_occurances)
print('Min occurrences:', min_occurances)
print('=====================================================================')

for class_info in class_infos:
    # Get one_hot representation of current class
    hot_class = one_hot.encode(class_info['class'], classes)
    # Calculate how many new samples have to be generated
    how_many_gen = int(round(class_info['deviation'] * b_ratio))
    print('\nClass: {}; How many new samples to generate: {}'.format(class_info['class'], how_many_gen))
    # Create images and labels for data representing current class
    images = np.asarray([train_rec['features'].reshape((box_size, box_size, 1)) for train_rec in train if np.array_equal(train_rec['label'], hot_class)])
    labels = np.tile(hot_class, reps=(class_info['occurrences'], 1))

    # Generate new images
    # datagen.fit(images)
    new_data = []
    for X_batch, y_batch in datagen.flow(images, labels, batch_size=batch_size):
        # If enough samples were generated
        if len(new_data) >= how_many_gen:
            break;
        for idx in range(len(X_batch)):
            new_record = {'features': X_batch[idx].flatten(), 'label': y_batch[idx]}
            new_data.append(new_record)
            # plt.imshow(X_batch[0].reshape((box_size, box_size)), cmap='gray')
            # plt.show()

    print('CLASS: {}; NEW records: {};'.format(class_info['class'], len(new_data)))
    # Append newly generated data & shuffle given dataset
    train += new_data

# Shuffle sets
print('\nShuffling training set ...')
shuffle(train)

print('\nNEW Training set size:', len(train))

with open(os.path.join(train_out_dir, 'train.pickle'), 'wb') as f:
    pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Training data has been successfully dumped into', f.name)
with open(os.path.join(test_out_dir, 'test.pickle'), 'wb') as f:
    pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Testing data has been successfully dumped into', f.name)

print('\n\n# Like our facebook page @ https://www.facebook.com/mathocr/')
