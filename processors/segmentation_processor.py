import random

import numpy as np
import cv2

import augmentations as AUG



class SegmentationProcessor:

    def __init__(self, batch_size, num_classes, image_size,
                 enable_augmentation=True):
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._image_size = image_size
        self._enable_augmentation = enable_augmentation

    def _load_batch_data(self, batch_data):
        batch_image = []
        batch_segment = []
        for data in batch_data:
            image_path = data['image_path']
            image = cv2.imread(image_path)
            segment_path = data['segment_path']
            segment = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)
            segment = np.stack([(segment == i).astype(np.float32) \
                                for i in range(self._num_classes)])
            batch_image.append(image)
            batch_segment.append(segment)
        return batch_image, batch_segment

    def _augment(self, batch_image, batch_segment):
        batch_image = [cv2.resize(image, self._image_size)\
                       for image in batch_image]
        batch_segment = [np.stack([cv2.resize(s, self._image_size) \
                                  for s in segment]) \
                         for segment in batch_segment]
        return batch_image, batch_segment

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
            batch_image, batch_segment = self._load_batch_data(batch_data)
            batch_image, batch_segment = self._augment(batch_image,
                                                       batch_segment)
            batch_image = np.transpose(batch_image, (0, 3, 1, 2))
            queue.put({'data_id'   : [data['data_id'] for data in batch_data],
                       'image'     : np.array(batch_image).astype(np.float32),
                       'segment'   : np.array(batch_segment).astype(np.float32),
                       'worker_id' : worker_id,
                       'is_end'    : False if i < num_batches - 1 else True})
    
