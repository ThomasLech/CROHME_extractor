import os

import pickle
import matplotlib.pyplot as plt


'Constants'
outputs_rel_path = 'outputs'
train_dir = os.path.join(outputs_rel_path, 'train')
test_dir = os.path.join(outputs_rel_path, 'test')
validation_dir = os.path.join(outputs_rel_path, 'validation')




if __name__ == '__main__':

	'Load pickled data'
	with open(os.path.join(train_dir, 'train.pickle'), 'rb') as train:
		train_pixels = pickle.load(train)

	with open(os.path.join(test_dir, 'test.pickle'), 'rb') as test:
		test_pixels = pickle.load(test)

	with open(os.path.join(validation_dir, 'validation.pickle'), 'rb') as validation:
		validation_pixels = pickle.load(validation)


	'Extract only labels iterating through all entries in train and test sets'
	labels = [sample['label'] for sample in (train_pixels + test_pixels)]

	'Sort by label occurancies'
	labels_set = sorted(list(set(labels)), key=lambda label: labels.count(label))
	labels_occr = [{'label': label, 'count': labels.count(label)} for label in labels_set]

	'Dump labels with their samples counted'
	with open('label_histograms.txt', 'w') as desc:
		[desc.write(label_occr_enc['label'] + ' : ' + str(label_occr_enc['count']) + '\n') for label_occr_enc in labels_occr]



	'Plot histograms'
	occurancies = [label_occr_enc['count'] for label_occr_enc in labels_occr]
	label_indices = [label_idx for label_idx in range(len(labels_set))]

	plt.figure(figsize=(40, 3))  # width:20, height:3
	plt.bar(label_indices, occurancies, align='center', width=0.8)
	plt.xticks(label_indices, labels_set, size='small', rotation=65)

	plt.xlabel('class')
	plt.ylabel('numb of occurancies')
	plt.title('Distribution of classes across train, test sets')
	plt.legend()
	# plt.show()

	'Dump plot into image file'
	plt.savefig('label_histograms.png')