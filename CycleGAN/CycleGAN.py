from Generate_LowRes_Patches import *
import os
from PIL import Image
import cv2
import numpy as np
import math
import argparse
from scipy import ndimage, misc

parser = argparse.ArgumentParser()
parser.add_argument("--folder_LR", type=str, help = 'path to low_resolution WSI')
parser.add_argument("--folder_HR", type=str, help = 'where to output higher res WSI')
parser.add_argument("--resltoresl", type=str, help = 'lowerRestohigherRes e.g. 256to512')
parser.add_argument("--num_recursions", type=int, help = "logbase2 of high_res/256")
args = parser.parse_args()


#Must be formatted like AtoB_aaaa;00000;00000.jpg
def get_image_data(image_path):
	try:
		image_name = image_path.split('_')[1].split(';')[0]
		image_x_coord = (int)(image_path.split('_')[1].split(';')[1])
		image_y_coord = (int)(image_path.split('_')[1].split(';')[2].split('.')[0])
		return image_name, image_x_coord, image_y_coord
	except IndexError:
		print(image_path)

def generate_WSI_res(folder_LR, folder_HR, resltoresl, num_recursions):

	for image_name in os.listdir(folder_LR):
		#Skip the useless file
		if image_name == '.DS_Store':
			continue

		#Expand images
		current_image = Image_Class(os.path.join(folder_LR, image_name))
		current_image.compress(2)
		current_image.expand(2)

		#Generate patches off of this image
		directory = 'datasets/'+resltoresl+"/testA/"
		current_image.generate_patches_train(overlap_factor=1, window_size=256, image_name=image_name, 
			                  				 output_folder=directory, num_of_whitespace=0)

	#Run through CycleGAN
	os.system("CUDA_VISIBLE_DEVICES=1 python ../code/main.py --dataset_dir=" + resltoresl + " --phase=test --which_direction=AtoB")
	outputted_directory = 'test/'
	if not os.path.exists(folder_HR):
		os.mkdir(folder_HR)

	#Repiece generated crops
	current_image_name = "blank"
	current_image = 255*np.ones((int(256*math.pow(2, num_recursions)), int(256*math.pow(2, num_recursions)), 3))
	for each in sorted(os.listdir(outputted_directory)):
		if '.html' not in each:

			image_name, image_x_coord, image_y_coord = get_image_data(each)
			image_path = os.path.join(outputted_directory, each)
			patch = cv2.imread(image_path)

			#If we're on the same image as last iteration
			if image_name == current_image_name:
				#Place the image at these coords
				current_image[image_x_coord:image_x_coord+256, image_y_coord:image_y_coord+256, :] = patch
			else:
				#Save the previous and create a new one
				current_image = cv2.blur(current_image, (5, 5))
				imsave(os.path.join(folder_HR, current_image_name + '.jpg'), current_image)
				current_image = np.zeros((int(256*math.pow(2, num_recursions)), int(256*math.pow(2, num_recursions)), 3))

				#Update image name
				current_image_name = image_name

				#Place the image at these coords
				current_image[image_x_coord:image_x_coord+256, image_y_coord:image_y_coord+256, :] = patch
		else:
				os.remove(os.path.join(outputted_directory, each))
	

generate_WSI_res(args.folder_LR, args.folder_HR, args.resltoresl, args.num_recursions)
#Name images from pggan like aaaa.jpg aaab.jpg etc.
#Patches will be like aaaa;00000;00000.jpg
#Num recursions = log2(high_res/256)