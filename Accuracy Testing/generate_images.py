import os
import argparse
from image_class import image_class
from scipy.misc import imsave


# Generates new images half-half
def generate_more_images(input_folder, output_folder, num_images):
    # Counter for number of images
    counter = 1

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through input folder
    for image1 in os.listdir(input_folder):
        for image2 in os.listdir(input_folder):
            # Make sure images are different
            if image1 is image2:
                continue

            # Check number of images saved
            if counter > num_images:
                return

            # Read in the images
            image1_load = image_class(os.path.join(input_folder, image1))
            image2_load = image_class(os.path.join(input_folder, image2))

            # Combine the images
            result = image1_load.combine(image2_load)

            # Save the result
            imsave(os.path.join(output_folder, image1 + "__" + image2 + (str)(counter)), result)
            counter += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help="folder containing original images")
    parser.add_argument("--output_folder", type=str, help="folder to output mixed images")
    parser.add_argument("--num_images", type=int, help="maximum number of images to save")
    args = parser.parse_args()

    generate_more_images(args.input_folder, args.output_folder, args.num_images)
