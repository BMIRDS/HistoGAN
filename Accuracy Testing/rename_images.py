import argparse
import os
from scipy.misc import imsave
from image_class import image_class


def rename_images(input_folder, output_folder, starting_number):
    for i, image_name in enumerate(os.listdir(input_folder), starting_number):
        image_path = os.path.join(input_folder, image_name)
        os.rename(image_path, os.path.join(input_folder, "{}.png".format(i)))
    os.system("cp -r {}{} {}".format(input_folder, "*.png ", output_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        help="path to original folder with images")
    parser.add_argument("--output_folder", type=str,
                        help="where to output the newly named images")
    parser.add_argument("--starting_number", type=int, default=0,
                        help="what number to start at")
    args = parser.parse_args()
    rename_images(args.input_folder, args.output_folder, args.starting_number)
