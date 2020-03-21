import random

import numpy as np
import cv2

import augmentations as AUG



class ClassificationProcessor:

    def __init__(self, batch_size, num_classes, enable_augmentation=True):
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._enable_augmentation = enable_augmentation

    def _load_batch_data(self, batch_data):
        batch_image = []
        batch_target = []
        for data in batch_data:
            image_path = data['image_path']
            image = cv2.imread(image_path)
            target = data['label']
            target = np.eye(self._num_classes)[np.array(target)]
            batch_image.append(image)
            batch_target.append(target)
        return batch_image, batch_target

    def _augment(self, batch_image, batch_target):
        return batch_image, batch_target

    def __call__(self, dataset, queue, worker_id):
        split_dataset = []
        batch_dataset = []
        for data in dataset:
            batch_dataset.append(data)
            if len(batch_dataset) >= self._batch_size:
                split_dataset.append(batch_dataset)
                batch_dataset = []
        if len(batch_dataset) > 0:
            split_dataset.append(batch_dataset)
        dataset = split_dataset
        num_batches = len(dataset)
        """ loop for batch """
        for i, batch_data in enumerate(dataset, 0):
            batch_image, batch_target = self._load_batch_data(batch_data)
            batch_image, batch_target = self._augment(batch_image,
                                                     batch_target)
            batch_image = np.transpose(batch_image, (0, 3, 1, 2))
            queue.put({'data_id'   : [data['data_id'] for data in batch_data],
                       'image'     : np.array(batch_image).astype(np.float32),
                       'target'    : np.array(batch_target).astype(np.float32),
                       'worker_id' : worker_id,
                       'is_end'    : False if i < num_batches - 1 else True})
    
