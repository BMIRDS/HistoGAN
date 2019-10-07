from skimage.transform import rescale
from jasons_image_utils import *
import numpy as np
from scipy.misc import imsave
import cv2
import os
import math
import argparse


class Image_Class:

    def __init__(self, image_path):
        # Read in the image and its dimensions
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

    def get_image(self):
        return self.image

    def compress(self, factor):
        # Retain original width and height
        original_width = self.width
        original_height = self.height

        # Compress image and get new dimensions
        self.image = cv2.resize(self.image, None, fx=1.0/factor, fy=1.0/factor)
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

        print("Compressed image from size ", original_width, "x", original_height, " to ", self.width, "x", self.height)

    def expand(self, factor):
        # Retain original width and height
        original_width = self.width
        original_height = self.height

        # Resize image and get new dimensions
        self.image = cv2.resize(self.image, None, fx=factor, fy=factor)
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

        print("Expanded image from size ", original_width, "x", original_height, " to ", self.width, "x", self.height)

    def generate_patches_train(self, overlap_factor, window_size, image_name, output_folder, num_of_whitespace):

        # Create output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Pad it to the nearest multiple of window_size
        if self.width % window_size != 0 or self.height%window_size != 0:
            old_x, old_y = self.width, self.height
            new_x, new_y = self.width, self.height
            if self.height % window_size != 0:
                new_y = ((int)(self.height/window_size)+1)*window_size
            if self.width % window_size != 0:
                new_x = ((int)(self.width/window_size)+1)*window_size
            new_img = np.zeros((new_x, new_y, 3))
            new_img[:, :, :] = 255
            new_img[:old_x, :old_y, :] = self.image
            self.image = new_img
            self.width = self.image.shape[0]
            self.height = self.image.shape[1]

        # Number of starting points for x and y and step size (same for x and y)
        x_steps = int((self.width - window_size) / window_size * overlap_factor)
        y_steps = int((self.height - window_size) / window_size * overlap_factor)
        step_size = int(window_size / overlap_factor)

        counter = 0
        # Loop through the amount of sliding windows
        for i in range(x_steps+1):
            for j in range(y_steps+1):
                # Starting and ending pixels
                x_start, x_end = i*step_size, i*step_size+window_size
                y_start, y_end = j*step_size, j*step_size+window_size

                # Quality Assurance
                assert x_start >= 0; assert y_start >= 0; assert x_end <= self.width; assert y_end <= self.height

                # Get the patch from the image
                crop = self.image[x_start:x_end, y_start:y_end, :]
                assert crop.shape == (window_size, window_size, 3)

                # Save the patch if it's not whitespace
                if num_of_whitespace == 0:
                    out_path = os.path.join(output_folder, image_name.split('.')[0]+';'+add_zeros(str(x_start))+';'+add_zeros(str(y_start))+'.jpg')
                    cv2.imwrite(out_path, crop)
                else:
                    if counter%num_of_whitespace==0: 
                        out_path = os.path.join(output_folder, image_name.split('.')[0]+';'+add_zeros(str(x_start))+';'+add_zeros(str(y_start))+'.jpg')
                        cv2.imwrite(out_path, crop)
                    counter+=1

    # Given high res and low res images, creates patches with half high res and half low res 
    def generate_mixed_patches(self, overlap_factor, window_size, image_name, output_folder, num_of_whitespace, low_res_image):

        # Create output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Number of starting points for x and y and step size (same for x and y)
        x_steps = int((self.width - window_size) / window_size * overlap_factor)
        y_steps = int((self.height - window_size) / window_size * overlap_factor)
        step_size = int(window_size / overlap_factor)

        counter = 0
        # Loop through the amount of sliding windows
        for i in range(x_steps+1):
            for j in range(y_steps+1):
                # Starting and ending pixels
                x_start, x_end = i*step_size, i*step_size+window_size
                y_start, y_end = j*step_size, j*step_size+window_size

                # Quality Assurance
                assert x_start >= 0; assert y_start >= 0; assert x_end <= self.width; assert y_end <= self.height

                # Get the patch from the image
                crop = np.ones((window_size, window_size, 3))
                # Left half high res
                crop[:, 0:(int)(window_size/2), :] = self.image[x_start:x_end, y_start:y_start+(int)(window_size/2), :]
                # Right half low res
                crop[:, (int)(window_size/2):window_size, :] = low_res_image.get_image()[x_start:x_end, y_start+(int)(window_size/2):y_end, :]
                assert crop.shape == (window_size, window_size, 3)

                # Save the patch if it's not whitespace
                if num_of_whitespace == 0:
                    out_path = os.path.join(output_folder, image_name.split('.')[0]+';'+add_zeros(str(x_start))+';'+add_zeros(str(y_start))+'.jpg')
                    cv2.imwrite(out_path, crop)
                else:
                    if counter%num_of_whitespace==0: 
                        out_path = os.path.join(output_folder, image_name.split('.')[0]+';'+add_zeros(str(x_start))+';'+add_zeros(str(y_start))+'.jpg')
                        cv2.imwrite(out_path, crop)
                    counter+=1        


    def generate_corner_patches(self, overlap_factor, window_size, image_name, output_folder, num_of_whitespace, low_res_image):
        # Create output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Number of starting points for x and y and step size (same for x and y)
        x_steps = int((self.width - window_size) / window_size * overlap_factor)
        y_steps = int((self.height - window_size) / window_size * overlap_factor)
        step_size = int(window_size / overlap_factor)

        counter = 0
        # Loop through the amount of sliding windows
        for i in range(x_steps+1):
            for j in range(y_steps+1):
                # Starting and ending pixels
                x_start, x_end = i*step_size, i*step_size+window_size
                y_start, y_end = j*step_size, j*step_size+window_size

                # Quality Assurance
                assert x_start >= 0; assert y_start >= 0; assert x_end <= self.width; assert y_end <= self.height

                # Get the patch from the image
                crop = np.ones((window_size, window_size, 3))
                # First make the whole thing high res
                crop[:, :, :] = self.image[x_start:x_end, y_start:y_end, :]
                # Then update a corner with the low res
                crop[0:(int)(window_size/2), (int)(window_size/2):window_size, :] = low_res_image.get_image()[x_start:x_start+(int)(window_size/2), y_start+(int)(window_size/2):y_end, :]
                assert crop.shape == (window_size, window_size, 3)

                # Save the patch if it's not whitespace
                if num_of_whitespace == 0:
                    out_path = os.path.join(output_folder, image_name.split('.')[0]+';'+add_zeros(str(x_start))+';'+add_zeros(str(y_start))+'.jpg')
                    cv2.imwrite(out_path, crop)
                else:
                    if counter%num_of_whitespace==0: 
                        out_path = os.path.join(output_folder, image_name.split('.')[0]+';'+add_zeros(str(x_start))+';'+add_zeros(str(y_start))+'.jpg')
                        cv2.imwrite(out_path, crop)
                    counter+=1        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help="input folder of large window ")
    parser.add_argument("--window_size", type=int, default=256, help="side length of square sliding window")
    parser.add_argument("--input_size", type=int, help = "size of the original images")
    parser.add_argument("--compression_factor", type=int, help="how much the images should be compressed/expanded")
    parser.add_argument("--window_overlap", type=float, help="e.g. 1/3 overlap type 3")
    parser.add_argument("--window_output_folder", type=str, help="where to output the patches")
    args = parser.parse_args()
    assert args.input_folder is not None
    assert args.compression_factor is not None


    # # Create directory for the compressed/re-expanded images
    directory = str(args.window_size) + "_" + str(args.window_size) + "|" + str(args.input_size) + "_" + str(args.input_size) + "|" + str(args.compression_factor)
    if not os.path.exists(directory):
        os.mkdir(directory)
     
    ######################################### GENERATE SAME RES PATCHES #################################
    # Loop through images in the input folder
    for image_name in os.listdir(args.input_folder):
        # Skip the useless file
        if image_name == '.DS_Store':
            continue

        # Compress and re-expand, save into folder
        current_image = Image_Class(os.path.join(args.input_folder, image_name))
        if args.compression_factor != 1:
            current_image.compress(args.compression_factor)
            current_image.expand(args.compression_factor)
        imsave(os.path.join(directory, image_name), current_image.get_image())

        # Generate patches off of this image
        current_image.generate_patches_train(args.window_overlap, args.window_size, image_name, args.window_output_folder, 0)

    # ######################################### GENERATE DIFFERENT RES PATCHES #################################
    # for image_name in os.listdir(args.input_folder):
    #     # Skip useless file
    #     if image_name == '.DS_Store':
    #         continue

    #     # Uncompressed image
    #     high_res_image = Image_Class(os.path.join(args.input_folder, image_name))

    #     # Compressed image
    #     low_res_image = Image_Class(os.path.join(args.input_folder, image_name))
    #     low_res_image.compress(2)
    #     low_res_image.expand(2)

    #     # Generate patches
    #     high_res_image.generate_mixed_patches(args.window_overlap, args.window_size, image_name, args.window_output_folder, 1, low_res_image)


    # ######################################### GENERATE CORNER PATCHES #################################
    # for image_name in os.listdir(args.input_folder):
    #     # Skip useless file
    #     if image_name == '.DS_Store':
    #         continue

    #     # Uncompressed image
    #     high_res_image = Image_Class(os.path.join(args.input_folder, image_name))

    #     # Compressed image
    #     low_res_image = Image_Class(os.path.join(args.input_folder, image_name))
    #     low_res_image.compress(2)
    #     low_res_image.expand(2)

    #     # Generate patches
    #     high_res_image.generate_corner_patches(args.window_overlap, args.window_size, image_name, args.window_output_folder, 1, low_res_image)

