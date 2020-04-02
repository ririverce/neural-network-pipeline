import os
if __name__ == '__main__':
    os.sys.path.append('../')
import copy
import random

import numpy as np
import cv2

from processors.anchor_box_processor import AnchorBoxProcessor
import augmentations as AUG



class PascalVOCAnchorBoxProcessor(AnchorBoxProcessor):

    def __init__(self, batch_size, num_classes, default_box, image_size,
                 iou_threshold=0.5, enable_augmentation=True):
        super(PascalVOCAnchorBoxProcessor, self).__init__(batch_size,
                                                          num_classes,
                                                          default_box,
                                                          image_size,
                                                          iou_threshold,
                                                          enable_augmentation)

    def _augment(self, batch_images, batch_labels, batch_bboxes):
        output_batch_images = []
        output_batch_labels = []
        output_batch_bboxes = []
        for image, labels, bboxes in zip(batch_images, batch_labels, batch_bboxes):
            #if self._enable_augmentation:
            #    tmp_image, tmp_labels, tmp_bboxes = AUG.bbox_random_crop(image, labels, bboxes)
            #    while len(tmp_labels) == 0:
            #        tmp_image, tmp_labels, tmp_bboxes = AUG.bbox_random_crop(image, labels, bboxes)
            #    image, labels, bboxes = tmp_image, tmp_labels, tmp_bboxes
            image = cv2.resize(image, self._image_size)
            output_batch_images.append(image)
            output_batch_labels.append(labels)
            output_batch_bboxes.append(bboxes)
        return output_batch_images, output_batch_labels, output_batch_bboxes



def test():
    import os
    import tqdm
    import cv2
    import loaders
    import iterators
    import utils
    batch_size = 32
    num_classes = 21
    image_size = (300, 300)
    dataset = loaders.PascalVOCLoader('../datasets/PascalVOC').load()
    train_dataset, valid_dataset = dataset
    default_box = utils.anchor_box_utils.generate_default_box(
                      image_size,
                      utils.anchor_box_utils.num_grids_ssd300,
                      utils.anchor_box_utils.grid_step_ssd300,
                      utils.anchor_box_utils.grid_size_ssd300,
                      utils.anchor_box_utils.aspect_ratio_ssd300
                  )
    train_processor = PascalVOCAnchorBoxProcessor(
                          batch_size,
                          num_classes=num_classes,
                          default_box=default_box,
                          image_size=image_size,
                          enable_augmentation=True,
                      )
    train_iterator = iterators.MultiprocessIterator(train_dataset,
                                                    train_processor,
                                                    num_workers=2)
    for batch_data in tqdm.tqdm(train_iterator):
        batch_image = batch_data['image']
        batch_conf = batch_data['conf']
        batch_loc = batch_data['loc']
        for image, conf, loc in zip(batch_image, batch_conf, batch_loc):
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
            default_box_cxy = default_box[:, :2] 
            default_box_wh = default_box[:, 2:]
            loc_cxy = loc[:, :2]
            loc_wh = loc[:, 2:]
            bbox_cxy = loc_cxy * 0.1 * default_box_wh + default_box_cxy
            bbox_wh = np.exp(loc_wh * 0.2) * default_box_wh
            mask = np.max(loc, -1) > 0
            bbox_cxy = bbox_cxy[mask]
            bbox_wh = bbox_wh[mask]
            bbox_tl = bbox_cxy - bbox_wh / 2
            bbox_br = bbox_cxy + bbox_wh / 2
            bboxes = np.concatenate([bbox_tl, bbox_br], -1)
            height, width = image.shape[:2]
            for box in bboxes:
                x_min = int(box[0] * width)
                y_min = int(box[1] * height)
                x_max = int(box[2] * width)
                y_max = int(box[3] * height)
                image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                                      (0, 0, 255), 1)
            cv2.imshow('test', image)
            cv2.waitKey(0)


if __name__ == '__main__':
    test()