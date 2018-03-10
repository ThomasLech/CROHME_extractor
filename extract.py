# Use like:
# python extract.py -b 28 -d 2011 2012 2013 -c digits symbols -t 20

import os
import argparse
# PArse xml
import xml.etree.ElementTree as ET
import numpy as np
import cv2
# One-hot encoder/decoder
import one_hot
# Load / dump data
import pickle

data_dir = os.path.join('data', 'CROHME_full_v2')
# Construct the argument parse and parse the arguments
version_choices = ['2011', '2012', '2013']
# Load categories from `categories.txt` file
categories = [{'name': cat.split(':')[0], 'classes': cat.split(':')[1].split()} for cat in list(open('categories.txt', 'r'))]
category_names = [cat['name'] for cat in categories]

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
    return min(x_mins), min(y_mins), max(x_maxs) - min(x_mins), max(y_maxs) - min(y_mins)

def shift_trace_group(trace_grp, x_min, y_min):
    shifted_traces = []
    for trace in trace_grp['traces']:
        shifted_traces.append(np.subtract(trace, [x_min, y_min]))
    return {'label': trace_grp['label'], 'traces': shifted_traces}

def get_scale(width, height, box_size):
    ratio = width / height
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
    # 2. Rescale trace_group
    trace_group = rescale_trace_group(trace_group, width, height, box_size=box_size-thickness_pad*2)
    x, y, width_r, height_r = get_tracegrp_properties(trace_group)
    # Shift trace_group by thickness padding
    trace_group = shift_trace_group(trace_group, x_min=-thickness_pad, y_min=-thickness_pad)
    # Center inside square box (box_size X box_size)
    margin_x = (box_size - (width_r + thickness_pad*2)) // 2
    margin_y = (box_size - (height_r + thickness_pad*2)) // 2
    trace_group = shift_trace_group(trace_group, x_min=-margin_x, y_min=-margin_y)
    image = draw_trace(trace_group, box_size, thickness=thickness)
    # Get pattern's width & height
    pat_width, pat_height = width_r + thickness_pad*2, height_r + thickness_pad*2

    # ! TESTS
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    if width < box_size and height < box_size:
        raise Exception('Trace group is too small.')
    if x != 0 or y != 0:
        raise Exception('Trace group was inproperly shifted.')
    if pat_width == 0 or pat_height == 0:
        raise Exception('Some sides are 0 length.')
    if pat_width < box_size and pat_height < box_size:
        raise Exception('Both sides are < box_size.')
    if pat_width > box_size or pat_height > box_size:
        raise Exception('Some sides are > box_size.')
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
            # Flatten image & construct pattern object
            pattern = {'features': image.flatten(), 'label': label}
            train.append(pattern)
        except Exception as e:
            print(e)
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
            print(e)
            # Ignore damaged trace groups
            damaged += 1

# Sort classes alphabetically
classes = sorted(classes)
print('\nTraining set size:', len(train))
print('Testing set size:', len(test))
print('How many rejected trace groups:', damaged, '\n')

# Data POST-processing
# 1. Normalize features
# 2. Convert labels to one-hot format
for pat in train:
    pat['features'] = (pat['features'] / 255).astype(dtype=np.uint8)
    pat['label'] = one_hot.encode(pat['label'], classes)
for pat in test:
    pat['features'] = (pat['features'] / 255).astype(dtype=np.uint8)
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

print('\n# Like our facebook page @ https://www.facebook.com/mathocr/')
