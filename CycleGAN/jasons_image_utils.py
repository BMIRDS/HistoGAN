## Usage: a bunch of auxillary methods for image processing

## About: Jason Wei, jason.20@dartmouth.edu, last edited July 28, 2018







import time
import os
from os import listdir
from os.path import isfile, join, isdir

import argparse

import numpy as np
from random import randint

from scipy.misc import imsave
import skimage.measure
from skimage.transform import rescale, rotate

from PIL import Image
Image.MAX_IMAGE_PIXELS=1e10
import cv2

############################################
############################################
################# folders ##################
############################################
############################################

def get_image_names(folder):
	image_names = [f for f in listdir(folder) if isfile(join(folder, f))]
	if '.DS_Store' in image_names:
		image_names.remove('.DS_Store')
	return image_names

def get_image_paths(folder):
	image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
	if join(folder, '.DS_Store') in image_paths:
		image_paths.remove(join(folder, '.DS_Store'))
	return image_paths

def get_subfolder_paths(folder):
	subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
	if join(folder, '.DS_Store') in subfolder_paths:
		subfolder_paths.remove(join(folder, '.DS_Store'))
	return subfolder_paths

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

#return the size of a folder and the number of images in it
def get_folder_size_and_num_images(folder):
	image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
	if join(folder, '.DS_Store') in image_paths:
		image_paths.remove(join(folder, '.DS_Store'))
	file_size = 0
	for image_path in image_paths:
		file_size += os.path.getsize(image_path)
		#print(image_path, os.path.getsize(image_path))
	file_size_mb = file_size / 1000.0 / 1000.0
	return file_size_mb, len(image_paths)

############################################
############################################
################## images ##################
############################################
############################################

def add_zeros(string):
	while len(string) < 5:
		string = "0" + string
	return string

###########################################
########### FILTERING WHITESPACE ##########
###########################################

def is_purple_dot(r, g, b):
	rb_avg = (r+b)/2
	if r > g - 10 and b > g - 10 and rb_avg > g + 20:
		return True
	return False
	
#this is actually a better method than is whitespace, but only if your images are purple lols
def is_purple(crop):
	pooled = skimage.measure.block_reduce(crop, (int(crop.shape[0]/15), int(crop.shape[1]/15), 1), np.average)
	num_purple_squares = 0
	for x in range(pooled.shape[0]):
		for y in range(pooled.shape[1]):
			r = pooled[x, y, 0]
			g = pooled[x, y, 1]
			b = pooled[x, y, 2]
			if is_purple_dot(r, g, b):
				num_purple_squares += 1
	if num_purple_squares > 8: 
		return True
	return False

############################################
############################################
############### annotations ################
############################################
############################################

def print_labels_spreadsheet(val_crops_folder, output_csv, classes):

	whole_slide_name_to_classes = {}

	subfolder_paths = get_subfolder_paths(val_crops_folder)

	for subfolder_path in subfolder_paths:
		image_names = get_image_names(subfolder_path)
		for image_name in image_names:
			parts = image_name.split('_')
			label = parts[0]
			whole_slide_name = parts[1]
			if label in classes:
				if whole_slide_name not in whole_slide_name_to_classes:
					whole_slide_name_to_classes[whole_slide_name] = {label}
				else:
					whole_slide_name_to_classes[whole_slide_name].add(label)

	print(whole_slide_name_to_classes)

	writer = open(output_csv, 'w')
	writer.write("val_whole_slide,a,l,mp,p,s\n")
	for whole_slide_name in sorted(whole_slide_name_to_classes.keys()):
		out_line = whole_slide_name
		classes_in_whole_slide = whole_slide_name_to_classes[whole_slide_name]
		for _class in classes:
			if _class in classes_in_whole_slide:
				out_line += ",1"
			else:
				out_line += ",0"
		out_line += '\n'
		writer.write(out_line)
	writer.close()

# for testing
if __name__ == "__main__":

	val_crops_folder = "crops/big/ffpe/train"
	output_csv = "wsi/train_gen_labels.csv"
	classes = ['a', 'l', 'mp', 'p', 's', 'f', 'n']
	print_labels_spreadsheet(val_crops_folder, output_csv, classes)
	print("from annotations, labels in", output_csv)
	












