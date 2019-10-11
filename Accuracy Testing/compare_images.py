import scipy
from scipy.misc import imsave
import os
import cv2
import numpy as tf


# Removes items that appear in a listmore than once
def remove_duplicates(image_list):
    return list(set(image_list))


# Gets a list of all images (including if it's used multiple times)
def find_images(list_of_folders):
    length = len(list_of_folders)
    # Base case
    if length is 0:
        return []
    else:
        images_in_this_folder = []
        for image in os.listdir(list_of_folders[length-1]):
            images_in_this_folder.append(image)
        return images_in_this_folder + find_images(list_of_folders[:-1])


# Sees if an image string is in a folder
def image_in_folder(image, folder):
    return each in os.listdir(folder)


# Add border to image
def add_border(image, image_width, image_height, border_width):
    blank_image = tf.zeros((image_width, image_height, 3), tf.uint8)
    width, height = image.shape[:2]
    blank_image[border_width:border_width+width,
                border_width:border_width+height, :] = image
    return blank_image


def create_image_rows(list_of_folders, image_width, image_height,
                      output_folder, border_width, between_images):

    # List of unique images
    unique_images = remove_duplicates(find_images(list_of_folders))

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each unique image
    for image in unique_images:

        # Count the number of times this image appears
        counter = 0
        for folder in list_of_folders:
            if image_in_folder(image, folder):
                counter += 1

        if counter != len(list_of_folders):
            continue

        # Create an empty image
        blank_image = tf.ones((
            image_width,
            image_height*counter + (counter-1)*between_images + 2*between_images,
            3), tf.uint8)
        blank_image.fill(255)

        # Fill the empty image
        image_counter = 0
        for folder in list_of_folders:
            for image_name in os.listdir(folder):
                if image_name is image:
                    # Load the image
                    image_path = os.path.join(folder, image)
                    current_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                    current_image = add_border(current_image, image_width, image_height, border_width)

                    # Insert the image
                    height_start = image_height * image_counter
                    height_end = image_height * (image_counter+1)
                    if image_counter is 0:
                        margin = image_counter * between_images
                    else:
                        margin = (image_counter+2) * between_images
                    height_start += margin
                    height_end += margin

                    blank_image[:, height_start:height_end, :] = current_image
                    image_counter += 1

        # Save the image
        output_path = os.path.join(output_folder, image)
        imsave(output_path, blank_image)


if __name__ == "__main__":
    # Paths to folders containing the images to combine (Manaully change this, inputs of folder)
    input_folders = []
    # Name of the folder that will be created to save the combined images
    output_folder = "nototv_rows"
    # Width of the black border between images
    border_width = 3
    # Can also specify to include whitespace between images
    between_images = 40

    create_image_rows(
        list_of_folders=input_folders,
        image_width=256 + 2*border_width,
        image_height=256 + 2*border_width,
        output_folder=output_folder,
        border_width=border_width,
        between_images=between_images)
