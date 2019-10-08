import cv2
from scipy.misc import imsave
import argparse
import os
from image_class import image_class

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help="input path to folder containing images to be modified")
    parser.add_argument("--output_folder", type=str, help="folder name for resulting images")
    parser.add_argument("--compress", type=str, default='False', help="True or False - true = images will be compressed")
    parser.add_argument("--filter_dups", type=bool, default=True, help="True or False - true = duplicates will be removed")
    parser.add_argument("--add_AtoB", type=str, default='False', help="True or False - true = AtoB_ will be added in front")
    parser.add_argument("--compression_factor", type=float, default=1.0, help="e.g. convert 256 --> 224 should input (256/224), if not compressing put any number")
    parser.add_argument("--increase_brightness", type=str, default='False', help='True or False - true = increase the brightness of the pictures')
    args = parser.parse_args()

    # Useful for comparing CycleGAN images (CycleGAN code default adds AtoB_ in front of original image name)
    add_AtoB = False
    if args.add_AtoB is "true" or args.add_AtoB is "True":
        add_AtoB = True

    # Changing image dimensions
    compress = False
    if args.compress is 'true' or args.compress is "True":
        compress = True

    # Change image brightness
    brightness = False
    if args.increase_brightness is 'True' or args.increase_brightness is 'true':
        brightness = True

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for each in os.listdir(args.input_folder):

        if args.filter_dups is True and "dup" in each:
            continue

        if each.startswith(.) or '.html' in each:
            continue

        filepath = os.path.join(args.input_folder, each)

        current_image = image_class(filepath)

        if compress is True:
            current_image.compress(args.compression_factor)

        if brightness is True:
            current_image.increase_brightness(75)

        if add_AtoB is True:

            # This output path is specific to our project, if you are using our code you should specify your own paths

            # output_path = os.path.join(args.output_folder, "AtoB_" + each.split[:-4] + ".png")
            output_path = os.path.join(args.output_folder, "AtoB_" + each[:-4] + ".jpg")                                        # For real images

            imsave(output_path, current_image.get_image())
        else:

            # This output path is specific to our project, if you are using our code you should specify your own paths
            output_path = os.path.join(args.output_folder, each.split(".")[0] + ".png")                                         # For fake images
            # output_path = os.path.join(args.output_folder, each.split(".")[1][2:] + ".jpg")                                   # For real images

            current_image.save_image(output_path)
