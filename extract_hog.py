import sys
import os

import pickle

from skimage.feature import hog
import matplotlib.pyplot as plt

'constants'
outputs_rel_path = 'outputs'
train_dir = os.path.join(outputs_rel_path, 'train')
test_dir = os.path.join(outputs_rel_path, 'test')
validation_dir = os.path.join(outputs_rel_path, 'validation')



if __name__ == '__main__':

	'parse cmd input'
	print(' # Script flags:', '<hog_cell_size>', '\n')

	'parse 1st arg'
	if len(sys.argv) < 2:
		print('\n + Usage:', sys.argv[0], '<hog_cell_size>', '\n')
		exit()

	try:
		hog_cell_size = int(sys.argv[1])
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



	' **** HOG PARAMS **** '
	orientations = 8
	pixels_per_cell = (hog_cell_size, hog_cell_size)
	cells_per_block = (1, 1)



	' **** Extract hog features **** '

	' TRAIN SET '
	print('Extracting hog - TRAIN set ...')
	train_hog = []
	for pattern_enc in train_set[40:]:

		hog_enc = dict({'label': pattern_enc.get('label'), 'features': hog(pattern_enc.get('features'), \
						orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=False, \
						block_norm='L2-Hys')})
		train_hog.append(hog_enc)


	' TEST SET '
	print('Extracting hog - TEST set ...')
	test_hog = []
	for pattern_enc in test_set:

		hog_enc = dict({'label': pattern_enc.get('label'), 'features': hog(pattern_enc.get('features'), \
						orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=False, \
						block_norm='L2-Hys')})
		test_hog.append(hog_enc)


	' VALIDATION SET '
	print('Extracting hog - VALIDATION set ...')
	validation_hog = []
	for pattern_enc in validation_set:

		hog_enc = dict({'label': pattern_enc.get('label'), 'features': hog(pattern_enc.get('features'), \
						orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=False, \
						block_norm='L2-Hys')})
		validation_hog.append(hog_enc)




	' DUMP DATA '
	print('\nDumping extracted data ...')
	'Make dirs if needed'
	if not os.path.exists(train_dir):
		os.mkdir(train_dir)
	if not os.path.exists(test_dir):
		os.mkdir(test_dir)
	if not os.path.exists(validation_dir):
		os.mkdir(validation_dir)


	with open(os.path.join(train_dir, 'train_hog_' + str(hog_cell_size) + 'x' + str(hog_cell_size) + '.pickle'), 'wb') as train:
		pickle.dump(train_hog, train, protocol=pickle.HIGHEST_PROTOCOL)
		print('Data has been successfully dumped into', train.name)

	with open(os.path.join(test_dir, 'test_hog_' + str(hog_cell_size) + 'x' + str(hog_cell_size) + '.pickle'), 'wb') as test:
		pickle.dump(test_hog, test, protocol=pickle.HIGHEST_PROTOCOL)
		print('Data has been successfully dumped into', test.name)

	with open(os.path.join(validation_dir, 'validation_hog_' + str(hog_cell_size) + 'x' + str(hog_cell_size) + '.pickle'), 'wb') as validation:
		pickle.dump(validation_hog, validation, protocol=pickle.HIGHEST_PROTOCOL)
		print('Data has been successfully dumped into', validation.name)
