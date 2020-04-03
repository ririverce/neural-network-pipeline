import random
import copy

import numpy as np
import cv2



"""*************
***** Crop *****
*************"""
def segment_random_crop(image, segment, min_crop_size=(150, 150), prob=1.0):
    if random.random() > prob:
        return image, segment
    src_height, src_width = image.shape[:2]
    min_crop_width, min_crop_height = min_crop_size
    crop_x_min = random.randint(0, np.clip(src_width - min_crop_width, 0, None))
    crop_y_min = random.randint(0, np.clip(src_height - min_crop_height, 0, None))
    crop_x_max = random.randint(np.clip(crop_x_min + min_crop_width, None, src_width), 
                                src_width)
    crop_y_max = random.randint(np.clip(crop_y_min + min_crop_height, None, src_height),
                                src_height)
    image = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    segment = np.stack(
                  [s[crop_y_min:crop_y_max, crop_x_min:crop_x_max] for s in segment]
              )
    return image, segment