import copy
import random
import gc

import numpy as np
import cv2

from processors.classification_processor import ClassificationProcessor
import augmentations as AUG



class ImageNetClassificationProcessor(ClassificationProcessor):

    def __init__(self, batch_size, num_classes, enable_augmentation, image_size):
        super(ImageNetClassificationProcessor, self).__init__(batch_size,
                                                              num_classes, 
                                                              enable_augmentation)
        self._image_size = image_size

    def _augment(self, batch_image, batch_target):
        """
        if self._enable_augmentation:
            rand_args = [i for i in range(len(batch_image))]
            random.shuffle(rand_args)
            for i in range(len(batch_image)):
                image, image_mix = batch_image[i], batch_image[rand_args[i]]
                image = cv2.resize(image, self._image_size)
                image_mix = cv2.resize(image_mix, self._image_size)
                target, target_mix = batch_target[i], batch_target[rand_args[i]]
                image, target = AUG.random_mixup(image, image_mix,
                                                 target, target_mix)
                batch_image[i] = image
                batch_target[i] = target
        """
        output_batch_image = []
        output_batch_target = []
        for image, target in zip(batch_image, batch_target):
            image = cv2.resize(image, self._image_size)
            if self._enable_augmentation:
                image = AUG.random_shift(image)
                image = AUG.random_rotate(image)
                image = AUG.random_cutout(image)
            output_batch_image.append(image)
            output_batch_target.append(target)
        return output_batch_image, output_batch_target