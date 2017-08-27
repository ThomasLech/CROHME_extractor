import sys
import os

import pickle
import numpy as np


'constants'
outputs_rel_path = 'outputs'
train_dir = os.path.join(outputs_rel_path, 'train')
test_dir = os.path.join(outputs_rel_path, 'test')
validation_dir = os.path.join(outputs_rel_path, 'validation')



if __name__ == '__main__':

	'parse cmd input'
	print(' # Script flags:', '<hog_cell_size_1>', '<hog_cell_size_2>', '<hog_cell_size_3>', '...')

	'parse 1st arg'
	if len(sys.argv) < 2:
		print('\n + Usage:', sys.argv[0], '<hog_cell_size_1>', '<hog_cell_size_2>', '<hog_cell_size_3>', '...')
		exit()

	try:
		hog_cell_sizes = [hog_cell_size for hog_cell_size in sys.argv[1:]]
	except Exception as e:
		print(e)
		exit()


	'Load TRAIN HOGs'
	train_dif_hogs = []
	for hog_cell_size in hog_cell_sizes:

		with open(os.path.join(train_dir, 'train_hog_' + hog_cell_size + 'x' + hog_cell_size + '.pickle'), 'rb') as train:
			train_dif_hogs.append(pickle.load(train))

	train_phog_size = len(train_dif_hogs[0])
	train_phog = []
	for hog_enc_idx in range(train_phog_size):

		' **** MERGE all different hog representations of EACH PATTERN **** '
		PHOG_features = []
		for train_dif_hog in train_dif_hogs:

			PHOG_features += train_dif_hog[hog_enc_idx]['features'].tolist()


		PHOG_enc = dict({'label': train_dif_hogs[0][hog_enc_idx]['label'], 'features': np.asarray(PHOG_features)})
		train_phog.append(PHOG_enc)

	# ' **** MERGE all different hog representations of EACH PATTERN **** '
	# train_phog = [{'label': train_dif_hogs[0][hog_enc_idx]['label'], \
	# 			   'phog': np.asarray([train_dif_hog[hog_enc_idx]['hog'] for train_dif_hog in train_dif_hogs], dtype=np.float32)} \
	# 								for hog_enc_idx in range(len(train_dif_hogs[0]))]






	'Load TEST HOGs'
	test_dif_hogs = []
	for hog_cell_size in hog_cell_sizes:

		with open(os.path.join(test_dir, 'test_hog_' + hog_cell_size + 'x' + hog_cell_size + '.pickle'), 'rb') as test:
			test_dif_hogs.append(pickle.load(test))

	test_phog_size = len(test_dif_hogs[0])
	test_phog = []
	for hog_enc_idx in range(test_phog_size):

		' **** MERGE all different hog representations of EACH PATTERN **** '
		PHOG_features = []
		for test_dif_hog in test_dif_hogs:

			PHOG_features += test_dif_hog[hog_enc_idx]['features'].tolist()


		PHOG_enc = dict({'label': test_dif_hogs[0][hog_enc_idx]['label'], 'features': np.asarray(PHOG_features)})
		test_phog.append(PHOG_enc)





	'Load VALIDATION HOGs'
	validation_dif_hogs = []
	for hog_cell_size in hog_cell_sizes:

		with open(os.path.join(validation_dir, 'validation_hog_' + hog_cell_size + 'x' + hog_cell_size + '.pickle'), 'rb') as validation:
			validation_dif_hogs.append(pickle.load(validation))

	validation_phog_size = len(validation_dif_hogs[0])
	validation_phog = []
	for hog_enc_idx in range(validation_phog_size):

		' **** MERGE all different hog representations of EACH PATTERN **** '
		PHOG_features = []
		for validation_dif_hog in validation_dif_hogs:

			PHOG_features += validation_dif_hog[hog_enc_idx]['features'].tolist()


		PHOG_enc = dict({'label': validation_dif_hogs[0][hog_enc_idx]['label'], 'features': np.asarray(PHOG_features)})
		validation_phog.append(PHOG_enc)






	' DUMP DATA '
	print('\nDumping extracted data ...')


	phog_cell_sizes_str = ''
	for hog_cell_size in hog_cell_sizes:
		phog_cell_sizes_str += hog_cell_size + '_'

	phog_cell_sizes_str = phog_cell_sizes_str[:-1]		# Removes last redundant '_' separator


	with open(os.path.join(train_dir, 'train_phog_' + phog_cell_sizes_str + '.pickle'), 'wb') as train:
		pickle.dump(train_phog, train, protocol=pickle.HIGHEST_PROTOCOL)
		print('Data has been successfully dumped into', train.name)

	with open(os.path.join(test_dir, 'test_phog_' + phog_cell_sizes_str + '.pickle'), 'wb') as test:
		pickle.dump(test_phog, test, protocol=pickle.HIGHEST_PROTOCOL)
		print('Data has been successfully dumped into', test.name)

	with open(os.path.join(validation_dir, 'validation_phog_' + phog_cell_sizes_str + '.pickle'), 'wb') as validation:
		pickle.dump(validation_phog, validation, protocol=pickle.HIGHEST_PROTOCOL)
		print('Data has been successfully dumped into', validation.name)
