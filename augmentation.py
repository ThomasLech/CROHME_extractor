'''
This script partially balance classes using standard data augmentation techniques.
'''

import os
import argparse
import pickle

# from keras.preprocessing.image import ImageDataGenerator

# Scikit-image tools
from skimage import img_as_ubyte
from skimage import transform as tf
from skimage.morphology import thin

import cv2
import matplotlib.pyplot as plt
import numpy as np

import one_hot, contours

# Global variables
outputs_dir = 'outputs'
train_out_dir = os.path.join(outputs_dir, 'train')
test_out_dir = os.path.join(outputs_dir, 'test')
classes = open('classes.txt', 'r').read().split()
categories = list(open('categories.txt', 'r'))

category_names = [category.split(':')[0] for category in categories]
category_classes = [category.split(':')[1].split() for category in list(open('categories.txt', 'r'))]

box_size = 45

# print(classes)

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--category', nargs='+', required=True, help="Specify which data category should be expanded.", choices=category_names)
ap.add_argument('-a', '--angle', required=False, help="Specify angle by which images should be rotated.")
args = vars(ap.parse_args())

angle = 10
if args.get('angle', {}):
    angle = args.get('angle')
print('Skew angle:', angle)

def skew(image, angle):
    image = img_as_ubyte(image)
    # Make up some more space for pattern before skewing it
    placeholder = np.ones(shape=(3*box_size, 3*box_size), dtype=np.uint8) * 255
    # Place pattern in the middle of placeholder image
    x_offset=y_offset=15
    placeholder[y_offset: y_offset + image.shape[0], x_offset: x_offset + image.shape[1]] = image

    height, width = placeholder.shape[:2]
    center = (width // 2, height // 2)
    rot_Matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(placeholder, rot_Matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def process_skewed(skewed_img, thresh=190):
    # Threshold skewed image
    _, thresh = cv2.threshold(skewed_img, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
    # Crop image to extract pattern
    _, conts, _ = cv2.findContours(255 - thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # Process contour
    proc_visual, x_coord, y_coord, pat_width, pat_height = contours.process_contours(conts, box_size=box_size)
    # Thin image
    thinned_img = thin(1.0 - proc_visual)
    return 1.0 - thinned_img

# Load output data
with open(os.path.join(train_out_dir, 'train.pickle'), 'rb') as data:
    train_data = pickle.load(data)
with open(os.path.join(test_out_dir, 'test.pickle'), 'rb') as data:
    test_data = pickle.load(data)

print('Unbalanced TRAIN set size:', len(train_data))
print('Unbalanced TEST set size:', len(test_data))

generated_data = []
for category in args.get('category'):
    print(' * Category:', category)
    # Find category index
    category_idx = category_names.index(category)
    # Find classes belonging to category
    category_classes = categories[category_idx].split(':')[1].split()
    print('Category classes:', category_classes)

    # Extract records that represent classes belonging to category
    category_train = [train_record for train_record in train_data if one_hot.decode(train_record['label'], classes) in category_classes]
    category_test = [test_record for test_record in test_data if one_hot.decode(test_record['label'], classes) in category_classes]
    # print('X:', len(category_train))
    # print('Y:', len(category_test))

    # X_train = np.asarray([train_rec['features'].reshape((box_size, box_size, 1)) for train_rec in category_train])
    # y_train = np.asarray([one_hot.decode(train_rec['label'], classes) for train_rec in category_train])
    # # define data preparation
    # datagen = ImageDataGenerator(rotation_range=50)
    # # fit parameters from data
    # datagen.fit(X_train)
    # # configure batch size and retrieve one batch of images
    # for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=50):
    # 	# create a grid of 3x3 images
    # 	for i in range(0, 9):
    # 		plt.subplot(330 + 1 + i)
    # 		plt.imshow(X_batch[i].reshape(box_size, box_size), cmap=plt.get_cmap('gray'))
    # 	# show the plot
    # 	plt.show()
    # 	break



    # 1st data augmentation technique - SKEW
    for train_rec in category_train:
        image = train_rec.get('features').reshape((box_size, box_size))
        # Skew image in left and right direction
        skew_img1 = skew(image, float(angle))
        skew_img2 = skew(image, -float(angle))

        # Post process image to remove artifacts created by rotating image
        generated_img1 = process_skewed(skew_img1)
        generated_img2 = process_skewed(skew_img2)

        # Append new data entry
        generated_data.append({'features': generated_img1.flatten(), 'label': train_rec.get('label')})
        generated_data.append({'features': generated_img2.flatten(), 'label': train_rec.get('label')})

        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(image, cmap='gray')
        # axarr[0].set_title('Input image')
        # axarr[1].imshow(generated_img, cmap='gray')
        # axarr[1].set_title('Generated image')
        # plt.show()

    print('Generated dataset size:', len(generated_data))

    # Merge generated data with original dataset
    train_data += generated_data

print('Balanced TRAIN set size:', len(train_data))
print('Balanced TEST set size:', len(test_data))

# Pickle training data
with open(os.path.join(train_out_dir, 'train.pickle'), 'wb') as train:
    pickle.dump(train_data, train, protocol=pickle.HIGHEST_PROTOCOL)
    print('New data has been successfully dumped into', train.name)
