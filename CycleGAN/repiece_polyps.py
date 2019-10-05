import os
from PIL import Image
import cv2
import numpy as np
from scipy.misc import imsave
import math
import argparse
from scipy import ndimage, misc

parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, help = "path to images to be repieced")
parser.add_argument("--wsi_path", type=str, help = "path to the whole_slide_images")
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

def repiece_polyps(folder_path, wsi_path):
	#Repiece generated crops
	os.mkdir("repieced_predictions")
	current_image_name = "blank"
	current_image = 255*np.ones((1000, 1000, 3))
	for each in sorted(os.listdir(folder_path)):
		if '.html' not in each:

			image_name, image_x_coord, image_y_coord = get_image_data(each)
			image_path = os.path.join(folder_path, each)
			patch = cv2.imread(image_path)

			#If we're on the same image as last iteration
			if image_name == current_image_name:
				#Place the image at these coords
				current_image[image_x_coord:image_x_coord+256, image_y_coord:image_y_coord+256, :] = patch
			else:
				#Save the previous and create a new one
				# current_image = cv2.blur(current_image, (5, 5))
				imsave(os.path.join("repieced_predictions/", current_image_name + '.jpg'), current_image)
				wsi_image = cv2.imread(os.path.join(wsi_path, image_name + ".png"), cv2.IMREAD_UNCHANGED)
				
				#Pad it to the nearest multiple of window_size
				new_x, new_y = wsi_image.shape[0], wsi_image.shape[1]
				if wsi_image.shape[1] % 256 != 0:
					new_y = ((int)(wsi_image.shape[1]/256)+1)*256
				if wsi_image.shape[0] % 256 != 0:
					new_x = ((int)(wsi_image.shape[0]/256)+1)*256

				current_image = np.zeros((new_x, new_y, 3))

				#Update image name
				current_image_name = image_name

				#Place the image at these coords
				current_image[image_x_coord:image_x_coord+256, image_y_coord:image_y_coord+256, :] = patch
		else:
				os.remove(os.path.join(folder_path, each))



repiece_polyps(args.folder_path, args.wsi_path)