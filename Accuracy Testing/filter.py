import torch.nn as nn
import os
from os import listdir
from os.path import join, isfile, isdir
import torch
import operator
import argparse
import cv2
from torchvision import datasets, models, transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

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
#Outputs the n most confident images
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
	class_num_to_class = {i:get_classes("../../../jason/cp_clean/train_folder/train/")[i] for i in range(len(get_classes("../../../jason/cp_clean/train_folder/train/")))} 
	batch_num = 0

	correct_counter, total_counter = 0, 0

	for test_inputs, test_labels in dataloader:

		#Model predictions
		batch_window_names = window_names[batch_num*16:batch_num*16+16]
		test_inputs = test_inputs.to(device)
		test_outputs = active_model(test_inputs)
		softmax_test_outputs = nn.Softmax()(test_outputs)
		confidences, test_preds = torch.max(softmax_test_outputs, 1)

		for i in range(test_preds.shape[0]):
			if i >= len(batch_window_names):
				continue
			image_name = batch_window_names[i]
			confidence = confidences[i].data.item()
			predicted_class = class_num_to_class[test_preds[i].data.item()]

			#Check prediction
			if predicted_class == _class:
				path_results[image_name] = confidence
				correct_counter+=1
			elif predicted_class != _class and misclassified == True:
				output_path = os.path.join("misclassified/", predicted_class + "_" + image_name.split("/")[2])
				os.system("cp -r " + image_name + " " + output_path)

			total_counter += 1

		batch_num += 1

	sorted_results = sorted(path_results.items(), key=operator.itemgetter(1), reverse=True)

	# # To get all correctly predicted imagess
	# for i in range(len(sorted_results)):
	# 	if sorted_results[i][0].split('/')[2] not in os.listdir("results_orig/"):
	# 		original_path = sorted_results[i][0]
	# 		confidence = str(round(sorted_results[i][1], 3))
	# 		output_path = os.path.join(output_folder, original_path.split('/')[2])
	# 		os.system("cp -r " + original_path + " " + output_path)

	for i in range(n):
		original_path = sorted_results[i][0]
		confidence = str(round(sorted_results[i][1], 3))
		output_path = os.path.join(output_folder, confidence + "_" + original_path.split('/')[2])
		os.system("cp -r " + original_path + " " + output_path)

	print("---------------------------------------")
	print(round(1.0*correct_counter/total_counter, 3))



#Main 
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type = str, help="input path to folder to evaluate")
parser.add_argument("--top_image", type = int, help = "number of image to save")
parser.add_argument("--class_to_use", type = str, help = "class to test for")
parser.add_argument("--output_folder", type = str, help = "where to save top images")
parser.add_argument("--model_path", type = str, help = "path to model that should be used")
parser.add_argument("--save_misclassified", type = str, help = "whether to save the images that are misclassified")
args = parser.parse_args()

#Save images that are incorrectly classified in a separate folder?
save_misclassified = False
if args.save_misclassified == "True" or args.save_misclassified == "true":
	
	save_misclassified = True

filter_by_confidence(args.input_folder, args.model_path, args.top_image, args.output_folder, args.class_to_use, save_misclassified)







