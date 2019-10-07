import cv2
import os
import numpy as np
from scipy.misc import imsave

class image_class:

    def __init__(self, image_path):
        # Read in the image and its dimensions
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.original_path = image_path
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]

    def get_area(self):
        return (float)(self.width*self.height)

    def get_side_lengths(self):
        return self.width, self.height

    def get_size(self):
        return (int)(os.path.getsize(self.original_path))

    def get_image(self):
        return self.image

    def save_image(self, output_path):
        imsave(output_path, self.image)

    def increase_brightness(self, value):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        self.image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    def resize_to_square(self):
        # Get smallest dimension
        smaller_side = ""
        if self.width < self.height:
            smaller_side = "width"
        elif self.height < self.width:
            smaller_side = "height"
        else:
            smaller_side = "same"

        # Get the middle part of the image
        if smaller_side is "width":
            self.image = self.image[:, (int)((self.height-self.width)/2):(int)((self.height-self.width)/2 + self.width), :]
            self.height = self.image.shape[1]
        elif smaller_side is "height":
            start = (int)((self.width-self.height)/2)
            self.image = self.image[start:(int)(start+self.height), :, :]
            self.width = self.image.shape[0]

        print("Rescaled to " + str(self.width) + "x" + str(self.height))

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

    def compress_to_square(self, final_side_length):
        # Make the image a square
        self.resize_to_square()

        # Compress the image
        self.compress(self.width/final_side_length)

    def combine(self, other_image):
        # Check if images are compatible
        assert self.width is other_image.get_image().shape[0]
        assert self.height is other_image.get_image().shape[1]

        # Create a new image
        new_image = np.zeros((self.width, self.height, 3), np.uint8)

        # Fill in the image
        new_image[:, 0:int(self.height/2), :] = self.image[:, 0:int(self.height/2), :]
        new_image[:, int(self.height/2):, :] = other_image.get_image()[:, int(self.height/2):, :]

        return new_image
