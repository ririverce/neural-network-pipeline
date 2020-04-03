import random

import numpy as np
import cv2

from processors.segmentation_processor import SegmentationProcessor
import augmentations as AUG



class PascalVOCSegmentationProcessor(SegmentationProcessor):

    def __init__(self, batch_size, num_classes, image_size,
                 enable_augmentation=True):
        super(PascalVOCSegmentationProcessor, self).__init__(batch_size,
                                                             num_classes,
                                                             image_size,
                                                             enable_augmentation)

    def _augment(self, batch_image, batch_segment):
        out_batch_image = []
        out_batch_segment = []
        for image, segment in zip(batch_image, batch_segment):
            if self._enable_augmentation:
                image, segment = AUG.segment_random_crop(
                                     image,
                                     segment,
                                     min_crop_size=np.array(self._image_size)
                                 )
                image = AUG.random_contrast(image)
                image = AUG.random_hue(image, deg_range=(-45, 45))
            image = cv2.resize(image, self._image_size)
            segment = np.stack([cv2.resize(s, self._image_size) for s in segment])
            out_batch_image.append(image)
            out_batch_segment.append(segment)
        return out_batch_image, out_batch_segment