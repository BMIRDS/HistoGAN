import os
import random
import string
import skimage.measure
import csv
import numpy as np
import argparse
from image_class import image_class


# Returns a dictionary with new names like aa.jpg
def rename(shuffled_keys):
    # Create the dictionary - key is real image name (e.g. "real/no_alk_123.jpg"), the value is the new name to use (e.g. aa.png)
    result = {}

    # Add to the dictionary
    counter = 0
    for key in shuffled_keys:
        counter += 1
        result[key] = str(counter)

    return result


# Returns a list of shuffled keys of a dictionary
def get_shuffled_keys(dictionary):
    keys = list(dictionary.keys())
    random.shuffle(keys)
    return keys


# Outputs a folder of renamed images, csv's for pathologists to fill out and for real labels
def generate_turing_test(real_image_folder, synthetic_images_folder, output_folder, control_image_folder, baseline_1, baseline_2):
    # Dictionaries to record data
    real_or_fake_dictionary = {}                # the key is real image name (e.g. "real/no_alk_123.jpg"), the value is "R" or "F" or "C"
    original_and_renamed_dictionary = {}        # the key is real image name (e.g. "real/no_alk_123.jpg"), the value is the new name to use (e.g. aa.png)

    # Loop through real images and register them as real
    for image in os.listdir(real_image_folder):
        # Get image path
        image_path = os.path.join(real_image_folder, image)

        # Update dictionary
        real_or_fake_dictionary[image_path] = "R"

    # Loop through fake images and register them as fake
    for image in os.listdir(synthetic_images_folder):
        # Get image path
        image_path = os.path.join(synthetic_images_folder, image)

        # Update dictionary
        real_or_fake_dictionary[image_path] = "F"

    # Loop through control images folder and register them as control
    for image in os.listdir(control_image_folder):
        # Get image path
        image_path = os.path.join(control_image_folder, image)

        # Update dictionary
        real_or_fake_dictionary[image_path] = "C"

    # Get shuffled keys
    shuffled_keys = get_shuffled_keys(real_or_fake_dictionary)

    # Update to some new names
    original_and_renamed_dictionary = rename(shuffled_keys)

    # Write master csv
    with open('master_file.csv', mode='w') as file:
        # Prepare to write out file
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Original Image Name', 'New Name', 'Real or Model it came from'])

        # Write out each image
        for each in shuffled_keys:
            if '.DS_Store' not in each:
                writer.writerow([each, original_and_renamed_dictionary.get(each) + '.png', real_or_fake_dictionary.get(each)])

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write pathologist csv
    with open('pathologist_file.csv', mode='w') as file:
        # Prepare to write out file
        writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Image Name', 'Authenticity (R = Real or F = Fake)', 'Notes'])

        # Write out each image and save it to the output folder
        for each in shuffled_keys:
            writer.writerow([original_and_renamed_dictionary.get(each) + '.png', '', ''])

            # Save the image
            if '.DS_Store' not in each:
                os.system("cp -r " + each + " " + os.path.join(output_folder, original_and_renamed_dictionary.get(each) + '.png'))


# Whether an image is mostly whitespace, takes in np array
def is_whitespace(crop):
    pooled = skimage.measure.block_reduce(crop, (int(crop.shape[0]/10), int(crop.shape[1]/10), 3), np.average)
    pooled = np.mean(pooled, axis=2)
    pooled = np.rint(pooled[:9, :9])
    num_good_squares = 0
    for x in np.nditer(pooled):
        if x < 225:
            num_good_squares += 1
    if num_good_squares > 80:
        return False
    return True


# Gets N images from the folder (synthetic images should have the same name as real images they were generated from but be in different folder)
def get_best_images(input_folder_real, input_folder_synthetic, num_images, output_folder):
    # Count the number of images that have been saved
    counter = 1

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Gather synthetic image paths
    synthetic_images = os.listdir(input_folder_synthetic)
    random.shuffle(synthetic_images)

    # Loop through synthetic images
    for image in synthetic_images:
        # Make sure image is compatible
        if '.html' in image:
            continue

        # Full path of real image and synthetic image
        current_image_path_real = os.path.join(input_folder_real, image)
        current_image_path_fake = os.path.join(input_folder_synthetic, image)

        # Load images
        current_image_real = image_class(current_image_path_real)
        current_image_fake = image_class(current_image_path_fake)

        # Check the number of images that have been saved
        if counter > num_images:
            return

        current_image_fake.save_image(os.path.join(output_folder, image))
        counter += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_image_folder", type=str, help="path to the folder with real images")
    parser.add_argument("--fake_image_folder", type=str, help="path to the folder with fake images")
    parser.add_argument("--output_folder", type=str, help="where to output the renamed images")
    parser.add_argument("--num_images", type=int, help="N number of images to use per class (total of 2N images)")
    parser.add_argument("--operation_num", type=int, help="0 to generate Turing Test, 1 to get N images from folder")
    parser.add_argument("--control_image_folder", type=str, help="path to the folder with control images")
    parser.add_argument("--baseline_1_folder", type=str, help="path to the folder with first baseline images")
    parser.add_argument("--baseline_2_folder", type=str, help="path to the folder with second baseline images")
    args = parser.parse_args()

    if args.operation_num is 1:
        get_best_images(args.real_image_folder, args.fake_image_folder, args.num_images, args.output_folder)
    elif args.operation_num is 0:
        generate_turing_test(args.real_image_folder, args.fake_image_folder, args.output_folder, args.control_image_folder, args.baseline_1_folder, args.baseline_2_folder)
    else:
        print("Invalid Operation")
        exit()
