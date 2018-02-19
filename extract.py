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

class Extractor(object):

    """Extracts patterns from inkml files."""

    crohme_package = os.path.join('data', 'CROHME_full_v2')
    output_dir = 'outputs'

    versions_available = ['2011', '2012', '2013']

    # Loads all categories that are available
    def load_categories(self):

        with open('categories.txt', 'r') as desc:

            lines = desc.readlines()

            # Removing any whitespace characters appearing in the lines
            categories = [{ "name": line.split(":")[0],
                            "classes": line.split(":")[1].strip().split(" ")}
                            for line in lines]

            return categories

    def __init__(self, box_size, versions="2013", categories="all"):

        try:
            self.box_size = int(box_size)
        except ValueError:
            print("\n! Box size must be a number!\n")
            exit()

        # Load list of possibble categories
        self.categories_available = self.load_categories()

        # Split by '+' delimeters
        versions = versions.split('+')
        categories = categories.split('+')
        for version in versions:

            if version not in self.versions_available:

                print("\n! This dataset version does not exist!\n")
                exit()

        self.versions = versions

        # Get names of available categories
        category_names = [category["name"] for category in self.categories_available]
        classes = []
        for category in categories:

            if category in category_names:

                category_idx = category_names.index(category)
                # Get classes of corresponding category
                classes += self.categories_available[category_idx]["classes"]

            else:

                print("\n! This category does not exist!\n")
                print("# Possible categories:\n")
                [print(" ", category["name"]) for category in self.categories_available]
                exit()

        self.categories = categories
        self.classes = classes

        self.train_data = []
        self.test_data = []
        self.validation_data = []

    def pixels(self):

        # Load inkml files
        for version in self.versions:

            if version == "2011":
                data_dir = os.path.join(self.crohme_package, "CROHME2011_data")
                train_dir = os.path.join(data_dir, "CROHME_training")
                test_dir = os.path.join(data_dir, "CROHME_testGT")
                validation_dir = os.path.join(data_dir, "CROHME_test")

                self.train_data += self.parse_inkmls(train_dir)
                self.test_data += self.parse_inkmls(test_dir)
                self.validation_data += self.parse_inkmls(validation_dir)

            if version == "2012":
                data_dir = os.path.join(self.crohme_package, "CROHME2012_data")
                train_dir = os.path.join(data_dir, "trainData")
                test_dir = os.path.join(data_dir, "testDataGT")
                validation_dir = os.path.join(data_dir, "testData")

                self.train_data += self.parse_inkmls(train_dir)
                self.test_data += self.parse_inkmls(test_dir)
                self.validation_data += self.parse_inkmls(validation_dir)

            if version == "2013":
                data_dir = os.path.join(self.crohme_package, "CROHME2013_data")
                train_root_dir = os.path.join(data_dir, "TrainINKML")
                train_dir_1 = os.path.join(train_root_dir, "expressmatch")
                train_dir_2 = os.path.join(train_root_dir, "extension")
                train_dir_3 = os.path.join(train_root_dir, "HAMEX")
                train_dir_4 = os.path.join(train_root_dir, "KAIST")
                train_dir_5 = os.path.join(train_root_dir, "MathBrush")
                train_dir_6 = os.path.join(train_root_dir, "MfrDB")

                test_dir = os.path.join(data_dir, "TestINKMLGT")
                validation_dir = os.path.join(data_dir, "TestINKML")

                self.train_data += self.parse_inkmls(train_dir_1)
                self.train_data += self.parse_inkmls(train_dir_2)
                self.train_data += self.parse_inkmls(train_dir_3)
                self.train_data += self.parse_inkmls(train_dir_4)
                self.train_data += self.parse_inkmls(train_dir_5)
                self.train_data += self.parse_inkmls(train_dir_6)
                self.test_data += self.parse_inkmls(test_dir)
                self.validation_data += self.parse_inkmls(validation_dir)

        return self.train_data, self.test_data, self.validation_data

    def parse_inkmls(self, data_dir_abs_path):

        'Accumulates traces_data of all the inkml files\
        located in the specified directory'
        patterns_enc = []
        classes_rejected = []

        'Check object is a directory'
        if os.path.isdir(data_dir_abs_path):

            for inkml_file in os.listdir(data_dir_abs_path):

            	if inkml_file.endswith('.inkml'):
            		inkml_file_abs_path = os.path.join(data_dir_abs_path, inkml_file)

            		print('Parsing:', inkml_file_abs_path, '...')

            		' **** Each entry in traces_data represent SEPARATE pattern\
            			which might(NOT) have its label encoded along with traces that it\'s made up of **** '
            		traces_data_curr_inkml = self.get_traces_data(inkml_file_abs_path)

            		'Each entry in patterns_enc is a dictionary consisting of \
            		pattern_drawn matrix and its label'
            		ptrns_enc_inkml_curr, classes_rej_inkml_curr = self.convert_to_imgs(traces_data_curr_inkml, box_size=self.box_size)
            		patterns_enc += ptrns_enc_inkml_curr
            		classes_rejected += classes_rej_inkml_curr

        return patterns_enc

    def convert_to_imgs(self, traces_data, box_size):

        patterns_enc = []
        classes_rejected = []

        for pattern in traces_data:

            trace_group = pattern['trace_group']

            'mid coords needed to shift the pattern'
            min_x, min_y, max_x, max_y = self.get_min_coords(trace_group)

            'traceGroup dimensions'
            trace_grp_height, trace_grp_width = max_y - min_y, max_x - min_x

            'shift pattern to its relative position'
            shifted_trace_grp = self.shift_trace_grp(trace_group, min_x=min_x, min_y=min_y)

            'Interpolates a pattern so that it fits into a box with specified size'
            'method: LINEAR INTERPOLATION'
            try:
            	interpolated_trace_grp = self.interpolate(shifted_trace_grp, \
            										 trace_grp_height=trace_grp_height, trace_grp_width=trace_grp_width, box_size=self.box_size - 1)
            except Exception as e:
            	print(e)
            	print('This data is corrupted - skipping.')
            	classes_rejected.append(pattern.get('label'))

            	continue

            'Get min, max coords once again in order to center scaled patter inside the box'
            min_x, min_y, max_x, max_y = self.get_min_coords(interpolated_trace_grp)

            centered_trace_grp = self.center_pattern(interpolated_trace_grp, max_x=max_x, max_y=max_y, box_size=self.box_size)

            'Center scaled pattern so it fits a box with specified size'
            pattern_drawn = self.draw_pattern(centered_trace_grp, box_size=self.box_size)
            # Make sure that patterns are thinned (1 pixel thick)
            pat_thinned = 1.0 - thin(1.0 - np.asarray(pattern_drawn))
            # plt.imshow(pat_thinned, cmap='gray')
            # plt.show()
            pattern_enc = dict({'features': pat_thinned, 'label': pattern.get('label')})

            # Filter classes that belong to categories selected by the user
            if pattern_enc.get('label') in self.classes:

                patterns_enc.append(pattern_enc)

        return patterns_enc, classes_rejected

    # Extracting / parsing tools below
    def get_traces_data(self, inkml_file_abs_path):

    	traces_data = []

    	tree = ET.parse(inkml_file_abs_path)
    	root = tree.getroot()
    	doc_namespace = "{http://www.w3.org/2003/InkML}"

    	'Stores traces_all with their corresponding id'
    	traces_all = [{'id': trace_tag.get('id'),
    					'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
    								else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord.split(' ')] \
    							for coord in (trace_tag.text).replace('\n', '').split(',')]} \
    							for trace_tag in root.findall(doc_namespace + 'trace')]

    	'Sort traces_all list by id to make searching for references faster'
    	traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    	'Always 1st traceGroup is a redundant wrapper'
    	traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    	if traceGroupWrapper is not None:
    		for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

    			label = traceGroup.find(doc_namespace + 'annotation').text

    			'traces of the current traceGroup'
    			traces_curr = []
    			for traceView in traceGroup.findall(doc_namespace + 'traceView'):

    				'Id reference to specific trace tag corresponding to currently considered label'
    				traceDataRef = int(traceView.get('traceDataRef'))

    				'Each trace is represented by a list of coordinates to connect'
    				single_trace = traces_all[traceDataRef]['coords']
    				traces_curr.append(single_trace)


    			traces_data.append({'label': label, 'trace_group': traces_curr})

    	else:
    		'Consider Validation data that has no labels'
    		[traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]

    	return traces_data

    def get_min_coords(self, trace_group):

    	min_x_coords = []
    	min_y_coords = []
    	max_x_coords = []
    	max_y_coords = []

    	for trace in trace_group:

    		x_coords = [coord[0] for coord in trace]
    		y_coords = [coord[1] for coord in trace]

    		min_x_coords.append(min(x_coords))
    		min_y_coords.append(min(y_coords))
    		max_x_coords.append(max(x_coords))
    		max_y_coords.append(max(y_coords))

    	return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)

    'shift pattern to its relative position'
    def shift_trace_grp(self, trace_group, min_x, min_y):

    	shifted_trace_grp = []

    	for trace in trace_group:
    		shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in trace]

    		shifted_trace_grp.append(shifted_trace)

    	return shifted_trace_grp

    'Interpolates a pattern so that it fits into a box with specified size'
    def interpolate(self, trace_group, trace_grp_height, trace_grp_width, box_size):

    	interpolated_trace_grp = []

    	if trace_grp_height == 0:
    		trace_grp_height += 1
    	if trace_grp_width == 0:
    		trace_grp_width += 1

    	'' 'KEEP original size ratio' ''
    	trace_grp_ratio = (trace_grp_width) / (trace_grp_height)

    	scale_factor = 1.0
    	'' 'Set \"rescale coefficient\" magnitude' ''
    	if trace_grp_ratio < 1.0:

    		scale_factor = (box_size / trace_grp_height)
    	else:

    		scale_factor = (box_size / trace_grp_width)

    	for trace in trace_group:
    		'coordintes convertion to int type necessary'
    		interpolated_trace = [[round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in trace]

    		interpolated_trace_grp.append(interpolated_trace)

    	return interpolated_trace_grp

    def center_pattern(self, trace_group, max_x, max_y, box_size):

    	x_margin = int((box_size - max_x) / 2)
    	y_margin = int((box_size - max_y) / 2)

    	return self.shift_trace_grp(trace_group, min_x= -x_margin, min_y= -y_margin)

    def draw_pattern(self, trace_group, box_size):

    	pattern_drawn = np.ones(shape=(box_size, box_size), dtype=np.float32)
    	for trace in trace_group:

    		' SINGLE POINT TO DRAW '
    		if len(trace) == 1:
    			x_coord = trace[0][0]
    			y_coord = trace[0][1]
    			pattern_drawn[y_coord, x_coord] = 0.0

    		else:
    			' TRACE HAS MORE THAN 1 POINT '

    			'Iterate through list of traces endpoints'
    			for pt_idx in range(len(trace) - 1):

    				'Indices of pixels that belong to the line. May be used to directly index into an array'
    				pattern_drawn[line(r0=trace[pt_idx][1], c0=trace[pt_idx][0],
    								   r1=trace[pt_idx + 1][1], c1=trace[pt_idx + 1][0])] = 0.0

    	return pattern_drawn

if __name__ == '__main__':

    out_formats = ['pixels', 'hog', 'phog']

    if len(sys.argv) < 3:

        print("\n! Usage:", "python", sys.argv[0], "<out_format>", "<box_size>", "<dataset_version=2013>", "<category=all>\n")
        exit()

    elif len(sys.argv) >= 3:

        if sys.argv[1] in out_formats:

            out_format = sys.argv[1]
            extractor = Extractor(sys.argv[2])
        else:

            print("\n! This output format does not exist!\n")
            print("# Possible output formats:\n")
            [print(" ", out_format) for out_format in out_formats]
            exit()
        if len(sys.argv) == 4:
            extractor = Extractor(sys.argv[2], sys.argv[3])
        elif len(sys.argv) == 5:
            extractor = Extractor(sys.argv[2], sys.argv[3], sys.argv[4])

    # Extract pixel features
    if out_format == out_formats[0]:

        train_data, test_data, validation_data = extractor.pixels()
        # Get list of all classes
        classes = sorted(list(set([data_record['label'] for data_record in train_data+test_data])))
        print('\nHow many classes:', len(classes))
        print('How many training samples:', len(train_data))
        print('How many testing samples:', len(test_data))
        with open('classes.txt', 'w') as desc:
            for r_class in classes:
                desc.write(r_class + '\n')
        # 1. Flatten image to single feaute map (vector of pixel intensities)
        # 2. Convert its label to one-hot format
        train_data = [{'label': one_hot.encode(train_rec['label'], classes), 'features': train_rec['features'].flatten()} for train_rec in train_data]
        test_data = [{'label': one_hot.encode(test_rec['label'], classes), 'features': test_rec['features'].flatten()} for test_rec in test_data]
        validation_data = [{'label': one_hot.encode(validation_rec['label'], classes), 'features': validation_rec['features'].flatten()} for validation_rec in validation_data]

    # Extract HOG features
    elif out_format == out_formats[1]:
        train_data, test_data, validation_data = extractor.hog()

    # Extract PHOG features
    elif out_format == out_formats[2]:
        train_data, test_data, validation_data = extractor.phog()

    output_dir = os.path.abspath(extractor.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_out_dir = os.path.join(output_dir, 'train')
    test_out_dir = os.path.join(output_dir, 'test')
    validation_out_dir = os.path.join(output_dir, 'validation')

    # Save data
    print('\nDumping extracted data ...')
    # Make directories if needed
    if not os.path.exists(train_out_dir):
        os.mkdir(train_out_dir)
    if not os.path.exists(test_out_dir):
        os.mkdir(test_out_dir)
    if not os.path.exists(validation_out_dir):
        os.mkdir(validation_out_dir)

    with open(os.path.join(train_out_dir, 'train.pickle'), 'wb') as train:
        pickle.dump(train_data, train, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data has been successfully dumped into', train.name)

    with open(os.path.join(test_out_dir, 'test.pickle'), 'wb') as test:
        pickle.dump(test_data, test, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data has been successfully dumped into', test.name)

    with open(os.path.join(validation_out_dir, 'validation.pickle'), 'wb') as validation:
        pickle.dump(validation_data, validation, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data has been successfully dumped into', validation.name)
