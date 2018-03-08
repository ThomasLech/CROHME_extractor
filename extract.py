import sys
import os

# Regex
import re

# PArse xml
import xml.etree.ElementTree as ET
import numpy as np
# Load / dump data
import pickle
# Draw line between two points
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt

# One-hot encoder/decoder
import one_hot





import argparse
import cv2

data_dir = os.path.join('data', 'CROHME_full_v2')
# Construct the argument parse and parse the arguments
version_choices = ['2011', '2012', '2013']
# Load categories from `categories.txt` file
categories = [{'name': cat.split(':')[0], 'classes': cat.split(':')[1].split()} for cat in list(open('categories.txt', 'r'))]
category_names = [cat['name'] for cat in categories]
# Use like:
# python extract.py -b 28 -d 2011 2012 2013 -c digits symbols -t 20

ap = argparse.ArgumentParser()
ap.add_argument('-b', '--box_size', required=True, help="Specify a length of square box side.")
ap.add_argument('-d', '--dataset_version', required=True, help="Specify what dataset versions have to be extracted.", choices=version_choices, nargs='+')
ap.add_argument('-c', '--category', required=True, help="Specify what dataset versions have to be extracted.", choices=category_names, nargs='+')
ap.add_argument('-t', '--thickness', required=False, help="Specify the thickness of extractd patterns.", default=1, type=int)
args = vars(ap.parse_args())
# Get classes that have to be extracted (based on categories selected by user)
classes_to_extract = []
for cat_name in args.get('category'):
    cat_idx = category_names.index(cat_name)
    classes_to_extract += categories[cat_idx]['classes']

# Extract INKML files
all_inkml_files = []
for d_version in args.get('dataset_version'):
    # Chose directory containing data based on dataset version selected
    working_dir = os.path.join(data_dir, 'CROHME{}_data'.format(d_version))
    # List folders found within working_dir
    for folder in os.listdir(working_dir):
        curr_folder = os.path.join(working_dir, folder)
        if os.path.isdir(curr_folder):
            # List files & folders found within folder
            content = os.listdir(curr_folder)
            # Filter inkml fiels and folders
            inkml_files = [os.path.join(curr_folder, inmkl_file) for inmkl_file in content if inmkl_file.endswith('.inkml')]
            sub_folders = [sub_folder for sub_folder in content if os.path.isdir(os.path.join(curr_folder, sub_folder))]

            print('FOLDER:', curr_folder)
            print('Numb. of inkml files:', len(inkml_files))

            all_inkml_files += inkml_files
            for sub_folder in sub_folders:
                # Extract inkml files from within sub_folder
                sub_folder_path = os.path.join(curr_folder, sub_folder)
                inkml_files = [os.path.join(sub_folder_path, inmkl_file) for inmkl_file in os.listdir(sub_folder_path) if inmkl_file.endswith('.inkml')]
                all_inkml_files += inkml_files

                print('FOLDER:', sub_folder_path)
                print('Numb. of inkml files:', len(inkml_files))
    print('\n')

# Filter inkml files that are used for training and those used for testing
training_inkmls = [inkml_file for inkml_file in all_inkml_files if 'CROHME_training' in inkml_file or 'trainData' in inkml_file or 'TrainINKML' in inkml_file]
testing_inkmls = [inkml_file for inkml_file in all_inkml_files if 'CROHME_testGT' in inkml_file or 'testDataGT' in inkml_file or ('TestINKMLGT' in inkml_file and not 'Prime_in_row' in inkml_file)]
print('Numder of training INKML files:', len(training_inkmls))
print('Numder of testing INKML files:', len(testing_inkmls))

train_data = []
test_data = []
classes = []
def extract_trace_grps(inkml_file_abs_path):
    trace_grps = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

    # Find traceGroup wrapper - traceGroup wrapping important traceGroups
    traceGrpWrapper = root.findall(doc_namespace + 'traceGroup')[0]
    traceGroups = traceGrpWrapper.findall(doc_namespace + 'traceGroup')
    for traceGrp in traceGroups:
        latex_class = traceGrp.findall(doc_namespace + 'annotation')[0].text
        traceViews = traceGrp.findall(doc_namespace + 'traceView')
        # Get traceid of traces that refer to latex_class extracted above
        id_traces = [traceView.get('traceDataRef') for traceView in traceViews]
        # Construct pattern object
        trace_grp = {'label': latex_class, 'traces': []}

        # Find traces with referenced by latex_class
        traces = [trace for trace in root.findall(doc_namespace + 'trace') if trace.get('id') in id_traces]
        # Extract trace coords
        for idx, trace in enumerate(traces):
            coords = []
            for coord in trace.text.replace('\n', '').split(','):
                # Remove empty strings from coord list (e.g. ['', '-238', '-91'] -> [-238', '-91'])
                coord = list(filter(None, coord.split(' ')))
                # Unpack coordinates
                x, y = coord[:2]
                # print('{}, {}'.format(x, y))
                if not float(x).is_integer():
                    # Count decimal places of x coordinate
                    d_places = len(x.split('.')[-1])
                    # ! Get rid of decimal places (e.g. '13.5662' -> '135662')
                    # x = float(x) * (10 ** len(x.split('.')[-1]) + 1)
                    x = float(x) * 10000
                else:
                    x = float(x)
                if not float(y).is_integer():
                    # Count decimal places of y coordinate
                    d_places = len(y.split('.')[-1])
                    # ! Get rid of decimal places (e.g. '13.5662' -> '135662')
                    # y = float(y) * (10 ** len(y.split('.')[-1]) + 1)
                    y = float(y) * 10000
                else:
                    y = float(y)

                # Cast x & y coords to integer
                x, y = round(x), round(y)
                coords.append([x, y])
            trace_grp['traces'].append(coords)
        trace_grps.append(trace_grp)

        # print('Pattern: {};'.format(pattern))
    return trace_grps

def get_tracegrp_properties(trace_group):
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
    for trace in trace_group['traces']:

        x_min, y_min = np.amin(trace, axis=0)
        x_max, y_max = np.amax(trace, axis=0)
        x_mins.append(x_min)
        x_maxs.append(x_max)
        y_mins.append(y_min)
        y_maxs.append(y_max)
    # print('X_min: {}; Y_min: {}; X_max: {}; Y_max: {}'.format(min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)))
    return min(x_mins), min(y_mins), max(x_maxs) - min(x_mins) + 1, max(y_maxs) - min(y_mins) + 1

def shift_trace_group(trace_grp, x_min, y_min):
    shifted_traces = []
    for trace in trace_grp['traces']:
        shifted_traces.append(np.subtract(trace, [x_min, y_min]))
    return {'label': trace_grp['label'], 'traces': shifted_traces}

def get_scale(width, height, box_size):
    box_size = box_size - 1
    ratio = width / height
    if width == 0:
        width += 1
    if height == 0:
        height += 1

    if ratio < 1.0:
        return box_size / height
    else:
        return box_size / width

def rescale_trace_group(trace_grp, width, height, box_size):
    # Get scale - we will use this scale to interpolate trace_group so that it fits into (box_size X box_size) square box.
    scale = get_scale(width, height, box_size)
    rescaled_traces = []
    for trace in trace_grp['traces']:
        # Interpolate contour and round coordinate values to int type
        rescaled_trace = np.around(np.asarray(trace) * scale).astype(dtype=np.uint8)
        rescaled_traces.append(rescaled_trace)

    return {'label': trace_grp['label'], 'traces': rescaled_traces}

def draw_trace(trace_grp, box_size, thickness):
    placeholder = np.ones(shape=(box_size, box_size), dtype=np.uint8) * 255
    for trace in trace_grp['traces']:
        for coord_idx in range(1, len(trace)):
            cv2.line(placeholder, tuple(trace[coord_idx - 1]), tuple(trace[coord_idx]), color=(0), thickness=thickness)
    return placeholder

def convert_to_img(trace_group):
    # Extract command line arguments
    box_size = int(args.get('box_size'))
    thickness = int(args.get('thickness'))
    # Calculate Thickness Padding
    thickness_pad = (thickness - 1) // 2
    # Convert traces to np.array
    trace_group['traces'] = np.asarray(trace_group['traces'])
    # Get properies of a trace group
    x, y, width, height = get_tracegrp_properties(trace_group)

    # 1. Shift trace_group
    trace_group = shift_trace_group(trace_group, x_min=x, y_min=y)
    x, y, width, height = get_tracegrp_properties(trace_group)
    if width == 0 or height == 0:
        raise Exception('Some sides are 0 length.')

    # 2. Rescale trace_group
    trace_group = rescale_trace_group(trace_group, width, height, box_size=box_size-thickness_pad*2)
    _, _, rescaled_w, rescaled_h = get_tracegrp_properties(trace_group)

    if rescaled_w == 0 or rescaled_h == 0:
        raise Exception('Some sides are 0 length.')
    # if rescaled_w < box_size and rescaled_h < box_size:
    #     raise Exception('Both sides are < box_size - 1')
    if rescaled_w > box_size or rescaled_h > box_size:
        raise Exception('Some sides are > box_size')

    # Shift trace_group by thickness padding
    trace_group = shift_trace_group(trace_group, x_min=-thickness_pad, y_min=-thickness_pad)

    # Center inside square box (box_size X box_size)
    margin_x = (box_size - rescaled_w + thickness) // 2
    margin_y = (box_size - rescaled_h + thickness) // 2
    trace_group = shift_trace_group(trace_group, x_min=-margin_x, y_min=-margin_y)
    image = draw_trace(trace_group, box_size, thickness=thickness)
    return image

damaged = 0
# Extract TRAINING data
train = []
for training_inkml in training_inkmls:
    print(training_inkml)
    trace_groups = extract_trace_grps(training_inkml)
    for trace_grp in trace_groups:
        label = trace_grp['label']
        # Extract only classes selected by user (selecting categories)
        if label not in classes_to_extract:
            continue
        try:
            if label not in classes:
                classes.append(label)
            # Convert patterns to images
            image = convert_to_img(trace_grp)
            # print(image)
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            # Flatten image & construct pattern object
            pattern = {'features': image.flatten(), 'label': label}
            train.append(pattern)
        except Exception as e:
            # print(e)
            # Ignore damaged trace groups
            damaged += 1

# Extract TESTING data
test = []
for testing_inkml in testing_inkmls:
    print(testing_inkml)
    trace_groups = extract_trace_grps(testing_inkml)
    for trace_grp in trace_groups:
        label = trace_grp['label']
        # Extract only classes selected by user (selecting categories)
        if label not in classes_to_extract:
            continue
        try:
            if label not in classes:
                classes.append(label)
            # Convert patterns to images
            image = convert_to_img(trace_grp)
            # Flatten image & construct pattern object
            pattern = {'features': image.flatten(), 'label': label}
            test.append(pattern)
        except Exception as e:
            # print(e)
            # Ignore damaged trace groups
            damaged += 1

# Sort classes alphabetically
classes = sorted(classes)
print('Training set size:', len(train))
print('Testing set size:', len(test))
print('How many damamged trace groups:', damaged)

# Data POST-processing
# 1. Normalize features
# 2. Convert labels to one-hot format
for pat in train:
    pat['features'] = pat['features'] / 255
    pat['label'] = one_hot.encode(pat['label'], classes)
for pat in test:
    pat['features'] = pat['features'] / 255
    pat['label'] = one_hot.encode(pat['label'], classes)

# Dump extracted data
outputs_dir = 'outputs'
train_out_dir = os.path.join(outputs_dir, 'train')
test_out_dir = os.path.join(outputs_dir, 'test')
# Make directories if needed
if not os.path.exists(outputs_dir):
    os.mkdir(outputs_dir)
if not os.path.exists(train_out_dir):
    os.mkdir(train_out_dir)
if not os.path.exists(test_out_dir):
    os.mkdir(test_out_dir)

with open(os.path.join(train_out_dir, 'train.pickle'), 'wb') as f:
    pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Training data has been successfully dumped into', f.name)
with open(os.path.join(test_out_dir, 'test.pickle'), 'wb') as f:
    pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Testing data has been successfully dumped into', f.name)
# Save all labels in 'classes.txt' file
with open('classes.txt', 'w') as f:
    for r_class in classes:
        f.write(r_class + '\n')
    print('All classes that were extracted are listed in {} file.'.format(f.name))

print('\n\n# Like our facebook page @ https://www.facebook.com/mathocr/')


# def get_traces_data(inkml_file_abs_path):
#     get_traces = []
#
#     tree = ET.parse(inkml_file_abs_path)
#     root = tree.getroot()
#     doc_namespace = "{http://www.w3.org/2003/InkML}"
#
#     # Iterate through trace tags (<trace>...</trace>)
#     for trace in root.findall(doc_namespace + 'trace'):
#         # Get all coords of current trace
#         for coord in trace_tag.text.replace('\n', '').split(','):
#             # Remove empty strings from coord list (e.g. ['', '-238', '-91'] -> [-238', '-91'])
#             coord = filter(None, coord)
#             print(coord)


	# 'Stores traces_all with their corresponding id'
	# traces_all = [{'id': trace_tag.get('id'),
	# 				'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
	# 								for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
	# 							else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
	# 								for axis_coord in coord.split(' ')] \
	# 						for coord in (trace_tag.text).replace('\n', '').split(',')]} \
	# 						for trace_tag in root.findall(doc_namespace + 'trace')]
    #
	# 'Sort traces_all list by id to make searching for references faster'
	# traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
	# 'Always 1st traceGroup is a redundant wrapper'
	# traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
    #
	# if traceGroupWrapper is not None:
	# 	for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
    #
	# 		label = traceGroup.find(doc_namespace + 'annotation').text
	# 		'traces of the current traceGroup'
	# 		traces_curr = []
	# 		for traceView in traceGroup.findall(doc_namespace + 'traceView'):
	# 			'Id reference to specific trace tag corresponding to currently considered label'
	# 			traceDataRef = int(traceView.get('traceDataRef'))
	# 			'Each trace is represented by a list of coordinates to connect'
	# 			single_trace = traces_all[traceDataRef]['coords']
	# 			traces_curr.append(single_trace)
    #
	# 		traces_data.append({'label': label, 'trace_group': traces_curr})
	# else:
	# 	'Consider Validation data that has no labels'
    #     for trace in traces_all:
    #         traces_data.append({'trace_group': [trace['coords']]})
    #
	# return traces_data


#
#
#
#
# class Extractor(object):
#
#     """Extracts patterns from inkml files."""
#
#     crohme_package = os.path.join('data', 'CROHME_full_v2')
#     output_dir = 'outputs'
#
#     versions_available = ['2011', '2012', '2013']
#
#     # Loads all categories that are available
#     def load_categories(self):
#
#         with open('categories.txt', 'r') as desc:
#
#             lines = desc.readlines()
#
#             # Removing any whitespace characters appearing in the lines
#             categories = [{ "name": line.split(":")[0],
#                             "classes": line.split(":")[1].strip().split(" ")}
#                             for line in lines]
#
#             return categories
#
#     def __init__(self, box_size, versions="2013", categories="all"):
#
#         try:
#             self.box_size = int(box_size)
#         except ValueError:
#             print("\n! Box size must be a number!\n")
#             exit()
#
#         # Load list of possibble categories
#         self.categories_available = self.load_categories()
#
#         # Split by '+' delimeters
#         versions = versions.split('+')
#         categories = categories.split('+')
#         for version in versions:
#
#             if version not in self.versions_available:
#
#                 print("\n! This dataset version does not exist!\n")
#                 exit()
#
#         self.versions = versions
#
#         # Get names of available categories
#         category_names = [category["name"] for category in self.categories_available]
#         classes = []
#         for category in categories:
#
#             if category in category_names:
#
#                 category_idx = category_names.index(category)
#                 # Get classes of corresponding category
#                 classes += self.categories_available[category_idx]["classes"]
#
#             else:
#
#                 print("\n! This category does not exist!\n")
#                 print("# Possible categories:\n")
#                 [print(" ", category["name"]) for category in self.categories_available]
#                 exit()
#
#         self.categories = categories
#         self.classes = classes
#
#         self.train_data = []
#         self.test_data = []
#         self.validation_data = []
#
#     def pixels(self):
#
#         # Load inkml files
#         for version in self.versions:
#
#             if version == "2011":
#                 data_dir = os.path.join(self.crohme_package, "CROHME2011_data")
#                 train_dir = os.path.join(data_dir, "CROHME_training")
#                 test_dir = os.path.join(data_dir, "CROHME_testGT")
#                 validation_dir = os.path.join(data_dir, "CROHME_test")
#
#                 self.train_data += self.parse_inkmls(train_dir)
#                 self.test_data += self.parse_inkmls(test_dir)
#                 self.validation_data += self.parse_inkmls(validation_dir)
#
#             if version == "2012":
#                 data_dir = os.path.join(self.crohme_package, "CROHME2012_data")
#                 train_dir = os.path.join(data_dir, "trainData")
#                 test_dir = os.path.join(data_dir, "testDataGT")
#                 validation_dir = os.path.join(data_dir, "testData")
#
#                 self.train_data += self.parse_inkmls(train_dir)
#                 self.test_data += self.parse_inkmls(test_dir)
#                 self.validation_data += self.parse_inkmls(validation_dir)
#
#             if version == "2013":
#                 data_dir = os.path.join(self.crohme_package, "CROHME2013_data")
#                 train_root_dir = os.path.join(data_dir, "TrainINKML")
#                 train_dir_1 = os.path.join(train_root_dir, "expressmatch")
#                 train_dir_2 = os.path.join(train_root_dir, "extension")
#                 train_dir_3 = os.path.join(train_root_dir, "HAMEX")
#                 train_dir_4 = os.path.join(train_root_dir, "KAIST")
#                 train_dir_5 = os.path.join(train_root_dir, "MathBrush")
#                 train_dir_6 = os.path.join(train_root_dir, "MfrDB")
#
#                 test_dir = os.path.join(data_dir, "TestINKMLGT")
#                 validation_dir = os.path.join(data_dir, "TestINKML")
#
#                 self.train_data += self.parse_inkmls(train_dir_1)
#                 self.train_data += self.parse_inkmls(train_dir_2)
#                 self.train_data += self.parse_inkmls(train_dir_3)
#                 self.train_data += self.parse_inkmls(train_dir_4)
#                 self.train_data += self.parse_inkmls(train_dir_5)
#                 self.train_data += self.parse_inkmls(train_dir_6)
#                 self.test_data += self.parse_inkmls(test_dir)
#                 self.validation_data += self.parse_inkmls(validation_dir)
#
#         return self.train_data, self.test_data, self.validation_data
#
#     def parse_inkmls(self, data_dir_abs_path):
#
#         'Accumulates traces_data of all the inkml files\
#         located in the specified directory'
#         patterns_enc = []
#         classes_rejected = []
#
#         'Check object is a directory'
#         if os.path.isdir(data_dir_abs_path):
#
#             for inkml_file in os.listdir(data_dir_abs_path):
#
#             	if inkml_file.endswith('.inkml'):
#             		inkml_file_abs_path = os.path.join(data_dir_abs_path, inkml_file)
#
#             		print('Parsing:', inkml_file_abs_path, '...')
#
#             		' **** Each entry in traces_data represent SEPARATE pattern\
#             			which might(NOT) have its label encoded along with traces that it\'s made up of **** '
#             		traces_data_curr_inkml = self.get_traces_data(inkml_file_abs_path)
#
#             		'Each entry in patterns_enc is a dictionary consisting of \
#             		pattern_drawn matrix and its label'
#             		ptrns_enc_inkml_curr, classes_rej_inkml_curr = self.convert_to_imgs(traces_data_curr_inkml, box_size=self.box_size)
#             		patterns_enc += ptrns_enc_inkml_curr
#             		classes_rejected += classes_rej_inkml_curr
#
#         return patterns_enc
#
#     def convert_to_imgs(self, traces_data, box_size):
#
#         patterns_enc = []
#         classes_rejected = []
#
#         for pattern in traces_data:
#
#             trace_group = pattern['trace_group']
#
#             'mid coords needed to shift the pattern'
#             min_x, min_y, max_x, max_y = self.get_min_coords(trace_group)
#
#             'traceGroup dimensions'
#             trace_grp_height, trace_grp_width = max_y - min_y, max_x - min_x
#
#             'shift pattern to its relative position'
#             shifted_trace_grp = self.shift_trace_grp(trace_group, min_x=min_x, min_y=min_y)
#
#             'Interpolates a pattern so that it fits into a box with specified size'
#             'method: LINEAR INTERPOLATION'
#             try:
#             	interpolated_trace_grp = self.interpolate(shifted_trace_grp, \
#             										 trace_grp_height=trace_grp_height, trace_grp_width=trace_grp_width, box_size=self.box_size - 1)
#             except Exception as e:
#             	print(e)
#             	print('This data is corrupted - skipping.')
#             	classes_rejected.append(pattern.get('label'))
#
#             	continue
#
#             'Get min, max coords once again in order to center scaled patter inside the box'
#             min_x, min_y, max_x, max_y = self.get_min_coords(interpolated_trace_grp)
#
#             centered_trace_grp = self.center_pattern(interpolated_trace_grp, max_x=max_x, max_y=max_y, box_size=self.box_size)
#
#             'Center scaled pattern so it fits a box with specified size'
#             pattern_drawn = self.draw_pattern(centered_trace_grp, box_size=self.box_size)
#             # Make sure that patterns are thinned (1 pixel thick)
#             pat_thinned = 1.0 - thin(1.0 - np.asarray(pattern_drawn))
#             # plt.imshow(pat_thinned, cmap='gray')
#             # plt.show()
#             pattern_enc = dict({'features': pat_thinned, 'label': pattern.get('label')})
#
#             # Filter classes that belong to categories selected by the user
#             if pattern_enc.get('label') in self.classes:
#
#                 patterns_enc.append(pattern_enc)
#
#         return patterns_enc, classes_rejected
#
#     # Extracting / parsing tools below
#     def get_traces_data(self, inkml_file_abs_path):
#
#     	traces_data = []
#
#     	tree = ET.parse(inkml_file_abs_path)
#     	root = tree.getroot()
#     	doc_namespace = "{http://www.w3.org/2003/InkML}"
#
#     	'Stores traces_all with their corresponding id'
#     	traces_all = [{'id': trace_tag.get('id'),
#     					'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
#     									for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
#     								else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
#     									for axis_coord in coord.split(' ')] \
#     							for coord in (trace_tag.text).replace('\n', '').split(',')]} \
#     							for trace_tag in root.findall(doc_namespace + 'trace')]
#
#     	'Sort traces_all list by id to make searching for references faster'
#     	traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
#
#     	'Always 1st traceGroup is a redundant wrapper'
#     	traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
#
#     	if traceGroupWrapper is not None:
#     		for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
#
#     			label = traceGroup.find(doc_namespace + 'annotation').text
#
#     			'traces of the current traceGroup'
#     			traces_curr = []
#     			for traceView in traceGroup.findall(doc_namespace + 'traceView'):
#
#     				'Id reference to specific trace tag corresponding to currently considered label'
#     				traceDataRef = int(traceView.get('traceDataRef'))
#
#     				'Each trace is represented by a list of coordinates to connect'
#     				single_trace = traces_all[traceDataRef]['coords']
#     				traces_curr.append(single_trace)
#
#
#     			traces_data.append({'label': label, 'trace_group': traces_curr})
#
#     	else:
#     		'Consider Validation data that has no labels'
#     		[traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]
#
#     	return traces_data
#
#     def get_min_coords(self, trace_group):
#
#     	min_x_coords = []
#     	min_y_coords = []
#     	max_x_coords = []
#     	max_y_coords = []
#
#     	for trace in trace_group:
#
#     		x_coords = [coord[0] for coord in trace]
#     		y_coords = [coord[1] for coord in trace]
#
#     		min_x_coords.append(min(x_coords))
#     		min_y_coords.append(min(y_coords))
#     		max_x_coords.append(max(x_coords))
#     		max_y_coords.append(max(y_coords))
#
#     	return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)
#
#     'shift pattern to its relative position'
#     def shift_trace_grp(self, trace_group, min_x, min_y):
#
#     	shifted_trace_grp = []
#
#     	for trace in trace_group:
#     		shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in trace]
#
#     		shifted_trace_grp.append(shifted_trace)
#
#     	return shifted_trace_grp
#
#     'Interpolates a pattern so that it fits into a box with specified size'
#     def interpolate(self, trace_group, trace_grp_height, trace_grp_width, box_size):
#
#     	interpolated_trace_grp = []
#
#     	if trace_grp_height == 0:
#     		trace_grp_height += 1
#     	if trace_grp_width == 0:
#     		trace_grp_width += 1
#
#     	'' 'KEEP original size ratio' ''
#     	trace_grp_ratio = (trace_grp_width) / (trace_grp_height)
#
#     	scale_factor = 1.0
#     	'' 'Set \"rescale coefficient\" magnitude' ''
#     	if trace_grp_ratio < 1.0:
#
#     		scale_factor = (box_size / trace_grp_height)
#     	else:
#
#     		scale_factor = (box_size / trace_grp_width)
#
#     	for trace in trace_group:
#     		'coordintes convertion to int type necessary'
#     		interpolated_trace = [[round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in trace]
#
#     		interpolated_trace_grp.append(interpolated_trace)
#
#     	return interpolated_trace_grp
#
#     def center_pattern(self, trace_group, max_x, max_y, box_size):
#
#     	x_margin = int((box_size - max_x) / 2)
#     	y_margin = int((box_size - max_y) / 2)
#
#     	return self.shift_trace_grp(trace_group, min_x= -x_margin, min_y= -y_margin)
#
#     def draw_pattern(self, trace_group, box_size):
#
#     	pattern_drawn = np.ones(shape=(box_size, box_size), dtype=np.float32)
#     	for trace in trace_group:
#
#     		' SINGLE POINT TO DRAW '
#     		if len(trace) == 1:
#     			x_coord = trace[0][0]
#     			y_coord = trace[0][1]
#     			pattern_drawn[y_coord, x_coord] = 0.0
#
#     		else:
#     			' TRACE HAS MORE THAN 1 POINT '
#
#     			'Iterate through list of traces endpoints'
#     			for pt_idx in range(len(trace) - 1):
#
#     				'Indices of pixels that belong to the line. May be used to directly index into an array'
#     				pattern_drawn[line(r0=trace[pt_idx][1], c0=trace[pt_idx][0],
#     								   r1=trace[pt_idx + 1][1], c1=trace[pt_idx + 1][0])] = 0.0
#
#     	return pattern_drawn
#
# if __name__ == '__main__':
#
#     out_formats = ['pixels', 'hog', 'phog']
#
#     if len(sys.argv) < 3:
#
#         print("\n! Usage:", "python", sys.argv[0], "<out_format>", "<box_size>", "<dataset_version=2013>", "<category=all>\n")
#         exit()
#
#     elif len(sys.argv) >= 3:
#
#         if sys.argv[1] in out_formats:
#
#             out_format = sys.argv[1]
#             extractor = Extractor(sys.argv[2])
#         else:
#
#             print("\n! This output format does not exist!\n")
#             print("# Possible output formats:\n")
#             [print(" ", out_format) for out_format in out_formats]
#             exit()
#         if len(sys.argv) == 4:
#             extractor = Extractor(sys.argv[2], sys.argv[3])
#         elif len(sys.argv) == 5:
#             extractor = Extractor(sys.argv[2], sys.argv[3], sys.argv[4])
#
#     # Extract pixel features
#     if out_format == out_formats[0]:
#
#         train_data, test_data, validation_data = extractor.pixels()
#         # Get list of all classes
#         classes = sorted(list(set([data_record['label'] for data_record in train_data+test_data])))
#         print('\nHow many classes:', len(classes))
#         print('How many training samples:', len(train_data))
#         print('How many testing samples:', len(test_data))
#         with open('classes.txt', 'w') as desc:
#             for r_class in classes:
#                 desc.write(r_class + '\n')
#         # 1. Flatten image to single feaute map (vector of pixel intensities)
#         # 2. Convert its label to one-hot format
#         train_data = [{'label': one_hot.encode(train_rec['label'], classes), 'features': train_rec['features'].flatten()} for train_rec in train_data]
#         test_data = [{'label': one_hot.encode(test_rec['label'], classes), 'features': test_rec['features'].flatten()} for test_rec in test_data]
#         validation_data = [{'label': one_hot.encode(validation_rec['label'], classes), 'features': validation_rec['features'].flatten()} for validation_rec in validation_data]
#
#     # Extract HOG features
#     elif out_format == out_formats[1]:
#         train_data, test_data, validation_data = extractor.hog()
#
#     # Extract PHOG features
#     elif out_format == out_formats[2]:
#         train_data, test_data, validation_data = extractor.phog()
#
#     output_dir = os.path.abspath(extractor.output_dir)
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#
#     train_out_dir = os.path.join(output_dir, 'train')
#     test_out_dir = os.path.join(output_dir, 'test')
#     validation_out_dir = os.path.join(output_dir, 'validation')
#
#     # Save data
#     print('\nDumping extracted data ...')
#     # Make directories if needed
#     if not os.path.exists(train_out_dir):
#         os.mkdir(train_out_dir)
#     if not os.path.exists(test_out_dir):
#         os.mkdir(test_out_dir)
#     if not os.path.exists(validation_out_dir):
#         os.mkdir(validation_out_dir)
#
#     with open(os.path.join(train_out_dir, 'train.pickle'), 'wb') as train:
#         pickle.dump(train_data, train, protocol=pickle.HIGHEST_PROTOCOL)
#         print('Data has been successfully dumped into', train.name)
#
#     with open(os.path.join(test_out_dir, 'test.pickle'), 'wb') as test:
#         pickle.dump(test_data, test, protocol=pickle.HIGHEST_PROTOCOL)
#         print('Data has been successfully dumped into', test.name)
#
#     with open(os.path.join(validation_out_dir, 'validation.pickle'), 'wb') as validation:
#         pickle.dump(validation_data, validation, protocol=pickle.HIGHEST_PROTOCOL)
#         print('Data has been successfully dumped into', validation.name)
