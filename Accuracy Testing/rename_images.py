import argparse
from scipy.misc import imsave
import os
from Image_Class import Image_Class

def rename_images(input_folder, output_folder, starting_number):

	for image_name in os.listdir(input_folder):

		image_path = os.path.join(input_folder, image_name)

		os.rename(image_path, os.path.join(input_folder, str(starting_number) + '.png'))

		# os.system("cp -r " + image_path + " " + os.path.join(output_folder, str(starting_number) + '.png'))

		# current_image = Image_Class(image_path)
		# current_image.save_image(os.path.join(output_folder, str(starting_number) + '.png'))

		starting_number += 1

	os.system("cp -r " + input_folder + "*.png " + output_folder)


parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type = str, help = "path to original folder with images")
parser.add_argument("--output_folder", type = str, help = "where to output the newly named images")
parser.add_argument("--starting_number", type = int, default = 0, help = "what number to start at")
args = parser.parse_args()

rename_images(args.input_folder, args.output_folder, args.starting_number)