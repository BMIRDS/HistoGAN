import argparse
import os
import cv2
from os.path import splitext
from scipy.misc import imsave
from image_class import image_class

"""Various functions that can be performed on images (e.g. compression, brightness)

PARAM CHANGES:
1. USE A BOOLEAN FLAG
--add_AtoB True -> --add_AtoB
--compress True -> --compress
--increase_brightness True -> --increase_brightness
--filter_dups False -> --no_filter_dups

"""


parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str,
                    help="input path to folder containing images to be modified")
parser.add_argument("--output_folder", type=str,
                    help="folder name for resulting images")
parser.add_argument("--compress", action="store_true", default=False,
                    help="boolean flag: true = images will be compressed\
                          (default): False")
parser.add_argument("--no_filter_dups", action="store_true", default=False,
                    help="boolean flag: true = duplicates will NOT be removed\
                          (default): False")
parser.add_argument("--add_AtoB", action="store_true", default=False,
                    help="boolean flag: true = AtoB_ will be added in front\
                          (default): False")
parser.add_argument("--compression_factor", type=float, default=1.0,
                    help="e.g. convert 256 --> 224 should input (256/224),\
                          if not compressing put any number")
parser.add_argument("--increase_brightness", action="store_true", default=False,
                    help='boolean flag: true = increase the brightness of the pictures\
                          (default): False')
args = parser.parse_args()

# Useful for comparing CycleGAN images (CycleGAN code default adds AtoB_ in front of original image name)
add_AtoB = args.add_AtoB

# Changing image dimensions
compress = args.compress

# Change image brightness
brightness = args.increase_brightness

os.makedirs(args.output_folder, exist_ok=True)

for each in os.listdir(args.input_folder):
    if args.no_filter_dups is False and "dup" in each:
        continue
    if each.startswith(.) or each.endswith('.html'):
        continue

    filepath = os.path.join(args.input_folder, each)
    current_image = image_class(filepath)
    if compress is True:
        current_image.compress(args.compression_factor)
    if brightness is True:
        current_image.increase_brightness(75)
    if add_AtoB is True:
        # For real images
        # This output path is specific to our project, if you are using
        # our code you should specify your own paths
        output_path = os.path.join(
            args.output_folder,
            "AtoB_{}.jpg".format(splitext(each)[0]))
        imsave(output_path, current_image.get_image())
    else:
        # For fake images
        # This output path is specific to our project, if you are using
        # our code you should specify your own paths
        output_path = os.path.join(
            args.output_folder,
            "{}.png".format(splitext(each)[0]))
        current_image.save_image(output_path)