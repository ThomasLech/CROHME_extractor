'''
This file contains some tools that are used in augmentation.py file.
'''

import numpy as np
import cv2

def shift_contour(contour, x_min, y_min):
	# Subtract (x_min, y_min) from every contour point
	return np.subtract(contour, [x_min, y_min])

def get_scale(cont_width, cont_height, box_size):
	box_size -= 1

	ratio = cont_width / cont_height
	if cont_width == 0:
		cont_width += 1
	if cont_height == 0:
		cont_height += 1

	if ratio < 1.0:
		return box_size / cont_height
	else:
		return box_size / cont_width

def process_contours(contours, box_size):
    # Initialize blank white box that will contain a single pattern
    # pattern = np.ones(shape=(h, w), dtype=np.uint8)
    visual = np.ones(shape=(box_size, box_size), dtype=np.uint8)
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    for cont in contours:
        # Make sure cont is a np.array
        # cont = np.asarray(cont)
        x_min, y_min = cont.min(axis=0)[0]
        x_max, y_max = cont.max(axis=0)[0]
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_mins.append(y_min)
        y_maxs.append(y_max)

    # Get global properties
    x_min = min(x_mins)
    x_max = max(x_maxs)
    y_min = min(y_mins)
    y_max = max(y_maxs)
    width = x_max - x_min
    height = y_max - y_min

    for cont in contours:
        # Shift contours
        shifted_cont = shift_contour(cont, x_min=x_min, y_min=y_min)
        # Get scale - we will use this scale to interpolate contour so that it fits into
        # box_size X box_size square box.
        scale = get_scale(width, height, box_size)
        # Interpolate contour and round coordinate values to int type
        rescaled_cont = (shifted_cont * scale).astype(dtype=np.uint8)
        # Get size of the rescaled contour
        rescaled_cont_width = width * scale
        rescaled_cont_height = height * scale
        # Get margin
        margin_x = int((box_size - rescaled_cont_width) / 2)
        margin_y = int((box_size - rescaled_cont_height) / 2)
        # Center pattern wihin a square box - we move pattern right by a proper margin
        centered_cont = np.add(rescaled_cont, [margin_x, margin_y])
        # Draw centered contour on a blank square box
        cv2.drawContours(visual, [centered_cont], contourIdx=0, color=(0))
    # Convert pattern's type to float
    visual = visual.astype(np.float)

    return visual, x_min, y_min, width, height
