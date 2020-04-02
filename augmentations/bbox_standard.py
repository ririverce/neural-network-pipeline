import random
import copy

import numpy as np
import cv2



"""*************
***** Crop *****
*************"""
def bbox_random_crop(image, labels, bboxes,
                     min_crop_size=(150, 150), threshold=0.1, prob=1.0):
    if random.random() > prob:
        return image, labels, bboxes
    src_height, src_width = image.shape[:2]
    image_area = src_width *src_height
    min_crop_width, min_crop_height = min_crop_size
    if src_width <= min_crop_width or src_height < min_crop_height:
        return image, labels, bboxes        
    crop_x_min = random.randint(0, src_width - min_crop_width)
    crop_y_min = random.randint(0, src_height - min_crop_height)
    crop_x_max = random.randint(crop_x_min, src_width)
    crop_y_max = random.randint(crop_y_min, src_height)
    image = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    crop_width = crop_x_max - crop_x_min
    crop_height = crop_y_max - crop_y_min
    crop_labels = []
    crop_bboxes = []
    for label, box in zip(labels, bboxes):
        src_box_area = (box[2] - box[0]) * (box[3] - box[1]) * image_area
        x_min = np.clip(0, box[0] * src_width - crop_x_min, crop_width)
        y_min = np.clip(0, box[1] * src_height - crop_y_min, crop_height)
        x_max = np.clip(0, box[2] * src_width - crop_x_min, crop_width)
        y_max = np.clip(0, box[3] * src_height - crop_y_min, crop_height)
        if (x_max - x_min) * (y_max - y_min) > threshold * src_box_area:
            x_min /= crop_width
            y_min /= crop_height
            x_max /= crop_width
            y_max /= crop_height
            crop_labels.append(label)
            crop_bboxes.append([x_min, y_min, x_max, y_max])
    labels = crop_labels
    bboxes = crop_bboxes
    return image, labels, bboxes