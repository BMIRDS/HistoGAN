import torch.nn as nn
import os
from os import listdir
from sklearn.metrics import roc_curve, auc, f1_score
from os.path import join, isfile, isdir
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import operator
import argparse
import cv2
from torchvision import datasets, models, transforms
from sklearn.metrics import roc_auc_score

random.seed(0)


#get full image paths
def get_image_paths(folder):
	image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
	if join(folder, '.DS_Store') in image_paths:
		image_paths.remove(join(folder, '.DS_Store'))
	image_paths = sorted(image_paths)
	return image_paths

#getting the classes for classification
def get_classes(folder):
	subfolder_paths = sorted([f for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)])
	return subfolder_paths

#Takes in a model and a folder of generated images
def filter_by_confidence(synthetic_folder, model, n, output_folder, _class, misclassified):

	#Set device for CUDA
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	if not os.path.exists("misclassified") and misclassified == True:
		os.makedirs("misclassified")

	#Load in the model
	active_model = torch.load(model)
	active_model.train(False)
	print("Loaded the model")

	#Results dictionary - key is image path, value is confidence
	path_results = {}

	#data transforms, no augmentation this time.
	data_transforms = {
		'normalize': transforms.Compose([
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15])
	    ]),
	    'unnormalize': transforms.Compose([
	        transforms.Normalize([1/0.15, 1/0.15, 1/0.15], [1/0.15, 1/0.15, 1/0.15])
	    ]),
	}

	#load the image dataset 
	image_dataset = datasets.ImageFolder(synthetic_folder, data_transforms['normalize']) #synthetic folder should be in a folder of same name (e.g. syn_tu/syn_tu/)
	dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=4)
	num_test_images = len(dataloader)*16

	window_names = get_image_paths(join(synthetic_folder, synthetic_folder))
	# class_num_to_class = {i:get_classes("../../../colorectal_polyps_ResNet/models/best1/train_folder/train/")[i] for i in range(len(get_classes("../../../colorectal_polyps_ResNet/models/best1/train_folder/train/")))} #5 classes
	class_num_to_class = {i:get_classes("../../../../jason/cp_clean/train_folder/train/")[i] for i in range(len(get_classes("../../../../jason/cp_clean/train_folder/train/")))} #6 classes
	batch_num = 0

	correct_counter, total_counter = 0, 0

	for test_inputs, test_labels in dataloader:

		#Model predictions
		batch_window_names = window_names[batch_num*16:batch_num*16+16]
		test_inputs = test_inputs.to(device)
		test_outputs = active_model(test_inputs)
		softmax_test_outputs = nn.Softmax()(test_outputs)
		confidences, test_preds = torch.max(softmax_test_outputs, 1)

		# print(test_preds)

		for i in range(test_preds.shape[0]):
			image_name = batch_window_names[i]
			confidence = confidences[i].data.item()
			predicted_class = class_num_to_class[test_preds[i].data.item()]

			# print(predicted_class, image_name, confidence)

			if predicted_class == _class:
				path_results[image_name] = confidence
				correct_counter+=1
			elif predicted_class != _class and misclassified == True:
				output_path = os.path.join("misclassified/", predicted_class + "_" + image_name.split("/")[2])
				os.system("cp -r " + image_name + " " + output_path)

			total_counter += 1

		batch_num += 1

	print("---------------------------------------")
	print(round(1.0*correct_counter/total_counter, 3))

	return correct_counter, total_counter

def roc(test_y, y_pred, ax, label_, color_):

	test_y_matrix = np.zeros((len(test_y), 2))
	pred_y_matrix = np.zeros((len(test_y), 2))

	for i in range(len(test_y_matrix)):
		test_y_matrix[i, test_y[i]] = 1
		pred_y_matrix[i, 0] = 1 - y_pred[i]
		pred_y_matrix[i, 1] = y_pred[i]

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(2):
		fpr[i], tpr[i], _ = roc_curve(test_y_matrix[:, i], pred_y_matrix[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	plt.rcParams.update({'font.size': 16})
	lw = 2
	ax.plot(fpr[1], tpr[1], color = color_, lw = lw, label = label_ + ' (AUC = %0.2f)' % roc_auc[0], linestyle='-')


#Takes in a model and a folder of generated images
def filter_by_confidence_binary(synthetic_folder, model, _class):

	#Set device for CUDA
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

	#Load in the model
	active_model = torch.load(model)
	active_model.train(False)
	print("Loaded the model")

	#data transforms, no augmentation this time.
	data_transforms = {
		'normalize': transforms.Compose([
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15])
	    ]),
	    'unnormalize': transforms.Compose([
	        transforms.Normalize([1/0.15, 1/0.15, 1/0.15], [1/0.15, 1/0.15, 1/0.15])
	    ]),
	}

	#load the image dataset 
	image_dataset = datasets.ImageFolder(synthetic_folder, data_transforms['normalize']) #synthetic folder should be in a folder of same name (e.g. syn_tu/syn_tu/)
	dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=4)
	num_test_images = len(dataloader)*16

	window_names = get_image_paths(join(synthetic_folder, synthetic_folder))
	class_num_to_class = {i:get_classes("../../../colorectal_polyps_ResNet/models/real/train_folder/train/")[i] for i in range(len(get_classes("../../../colorectal_polyps_ResNet/models/real/train_folder/train/")))} #6 classes
	batch_num = 0

	tp, fp, tn, fn = 0, 0, 0, 0
	binary_labels = []
	predicted_labels = []
	probabilities = []

	for test_inputs, test_labels in dataloader:

		#Model predictions
		batch_window_names = window_names[batch_num*16:batch_num*16+16]
		test_inputs = test_inputs.to(device)
		test_outputs = active_model(test_inputs)
		softmax_test_outputs = nn.Softmax()(test_outputs)
		confidences, test_preds = torch.max(softmax_test_outputs, 1)

		for i in range(test_preds.shape[0]):
			image_name = batch_window_names[i]
			confidence = confidences[i].data.item()
			predicted_class = class_num_to_class[test_preds[i].data.item()]

			#Replace our classes with your classes here
			if predicted_class == 'ss' and _class == 'ss':
				tp += 1
				binary_labels.append(1)
				predicted_labels.append(1)
				probabilities.append(confidence)
			elif predicted_class == 'ss' and _class == 'hp':
				fp += 1
				binary_labels.append(0)
				predicted_labels.append(1)
				probabilities.append(confidence)
			elif predicted_class == 'hp' and _class == 'hp':
				tn += 1
				binary_labels.append(0)
				predicted_labels.append(0)
				probabilities.append(1.0-confidence)
			elif predicted_class == 'hp' and _class == 'ss':
				fn += 1
				binary_labels.append(1)
				predicted_labels.append(0)
				probabilities.append(1.0-confidence)

		batch_num += 1

	return tp, fp, tn, fn, binary_labels, predicted_labels, probabilities


#Returns accuracy for multiple folders
def calculate_overall_accuracy(input_folders, input_classes, model_path, binary):
	correct_counter, total_counter = 0, 0

	tp, fp, tn, fn = 0, 0, 0, 0
	binary_labels = []
	predicted_labels = []
	probabilities = []

	#Loop through each folder 
	for i in range(len(input_folders)):
		if binary == True:
			#Model binary values
			class_tp, class_fp, class_tn, class_fn, class_binary, class_predicted, class_probabilities = filter_by_confidence_binary(synthetic_folder=input_folders[i], model=model_path, _class=input_classes[i])

			#Add to overall counter
			tp += class_tp
			fp += class_fp
			tn += class_tn
			fn += class_fn

			#Add in the predictions
			binary_labels = binary_labels + class_binary
			predicted_labels = predicted_labels + class_predicted
			probabilities = probabilities + class_probabilities
		elif binary == False:
			#Get model's accuracy on current folder
			class_correct, class_total = filter_by_confidence(synthetic_folder=input_folders[i], model=model_path, n=0, output_folder=input_folders[i], _class=input_classes[i], misclassified=False)

			#Add to overall counter
			correct_counter += class_correct
			total_counter += class_total

	if binary == False:
		print("----------------------------------------------------")
		print("Model: " + model_path)
		print("Accuracy: " + str(round(1.0*correct_counter/total_counter, 3)))
		print("----------------------------------------------------")
	elif binary == True:
		print("----------------------------------------------------")
		print("Model: " + model_path)
		print(tp, fp, tn, fn)
		print("Sensitivity: " + str(round(1.0*tp/(tp+fn), 3)))
		print("Specificity: " + str(round(1.0*tn/(tn+fp), 3)))
		# print("F1 Score: " + str(round(f1_score(binary_labels, predicted_labels), 3)))
		print("AUC: " + str(round(roc_auc_score(binary_labels, probabilities), 3)))
		print("----------------------------------------------------")

		# roc(binary_labels, probabilities)
		return binary_labels, probabilities


# MAIN #

input_folders = ['tu', 'tv', 'ss']	 														#Should be a list of folders where each folder contains images of a separate class
input_classes = ['tu', 'tv', 'ss']															#Classes' indexes should correspond with the folder they're with
model_paths = ['../models/real_resnet18_e20_va0.855.pt']									#Direct paths to models 
model_labels = ['DiscoGAN', 'DCGAN', 'CycleGAN']											#What to call each model
colors = ['turquoise', 'darkorange', 'violet']												#Colors for plotting
binary = False
do_roc = False

if do_roc == True:
	plt.figure()
	ax = plt.subplot()
	ax.tick_params(labelsize=14)

for i in range(len(model_paths)):
	model, label = model_paths[i], model_labels[i]

	binary_labels, probabilities = calculate_overall_accuracy(input_folders, input_classes, model, binary)

	if do_roc == True:
		roc(binary_labels, probabilities, ax, label, colors[i])

if do_roc == True:
	plt.rcParams.update({'font.size': 16})

	box = ax.get_position()

	plt.xlim([-0.02, 1.02])
	plt.ylim([-0.02, 1.02])
	plt.rcParams.update({'font.size': 16})
	plt.xlabel('False Positive Rate', fontsize=14)
	plt.ylabel('True Positive Rate', fontsize=14)
	# plt.title('Receiver Operating Characteristi	c', fontsize=14)
	ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])

	ax.legend(loc = 'lower right', fancybox=True, shadow=True, prop = {'size': 12}).get_frame().set_edgecolor('black')
	plt.savefig('AUROC.png', dpi=1500, format = 'png', bbox_inches='tight')











