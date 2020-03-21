import copy
import random

import numpy as np
import cv2

import augmentations as AUG



class AnchorBoxProcessor:

    def __init__(self, batch_size, num_classes, default_box, image_size,
                 iou_threshold=0.5, enable_augmentation=True):
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._default_box = default_box
        self._image_size = image_size
        self._iou_threshold = iou_threshold
        self._enable_augmentation = enable_augmentation

    def _load_batch_data(self, batch_data):
        batch_images = []
        batch_labels = []
        batch_bboxes = []
        for data in batch_data:
            image_path = data['image_path']
            image = cv2.imread(image_path)
            label = data['label']
            label = np.eye(self._num_classes)[np.array(label)]
            bbox = data['bbox']
            batch_images.append(image)
            batch_labels.append(label)
            batch_bboxes.append(bbox)
        return batch_images, batch_labels, batch_bboxes

    def _augment(self, batch_images, batch_labels, batch_bboxes):
        batch_images = [cv2.resize(image, self._image_size) for image in batch_images]
        return batch_images, batch_labels, batch_bboxes

    def _calc_iou_matrix(self, batch_bboxes):
        default_box_cxy = self._default_box[:, :2]
        default_box_wh = self._default_box[:, 2:]
        default_box_tl = default_box_cxy - (default_box_wh / 2.0)
        default_box_br = default_box_cxy + (default_box_wh / 2.0)
        default_box = np.concatenate([default_box_tl, default_box_br], axis=-1)
        iou_matrix_list = []
        for bboxes in batch_bboxes:
            iou_matrix = np.zeros([default_box.shape[0], len(bboxes)])
            for i, box in enumerate(bboxes, 0):
                tiled_box = np.tile([box], [default_box.shape[0], 1])
                tiled_box_tl = tiled_box[:, :2]
                tiled_box_br = tiled_box[:, 2:]
                tiled_box_wh = tiled_box_br - tiled_box_tl
                intersection_tl = np.maximum(default_box_tl, tiled_box_tl)
                intersection_br = np.minimum(default_box_br, tiled_box_br)
                intersection_wh = intersection_br - intersection_tl
                intersection_wh = np.clip(intersection_wh, 0, None)
                intersection_area = intersection_wh[:, 0] * intersection_wh[:, 1]
                default_box_area = default_box_wh[:, 0] * default_box_wh[:, 1]
                tiled_box_area = tiled_box_wh[:, 0] * tiled_box_wh[:, 1]
                union_area = default_box_area + tiled_box_area - intersection_area
                iou_matrix[:, i] = intersection_area / union_area
            iou_matrix_list.append(iou_matrix)
        return iou_matrix_list

    def _calc_matched_pair(self, iou_matrix_list):
        iou_matrix_list = copy.deepcopy(iou_matrix_list)
        matched_pair_list = []
        for iou_matrix in iou_matrix_list:
            matched_pair = []
            for _ in range(iou_matrix.shape[1]):
                max_args = [np.argmax(np.max(iou_matrix, axis=1)),
                            np.argmax(np.max(iou_matrix, axis=0))]
                iou_matrix[max_args[0]] = 0
                iou_matrix[:, max_args[1]] = 0
                matched_pair.append(max_args)
            matched_pair_list.append(matched_pair)
        return matched_pair_list
    
    def _remove_matched_iou(self, iou_matrix_list, matched_pair_list):
        for i, matched_pair in enumerate(matched_pair_list):
            for pair in matched_pair:
                iou_matrix_list[i][pair] = 0.0
        return iou_matrix_list

    def _calc_surrounding_pair(self, iou_matrix_list):
        matched_pair_list = []
        for iou_matrix in iou_matrix_list:
            matched_pair = []
            while np.max(iou_matrix) >= self._iou_threshold:
                max_args = [np.argmax(np.max(iou_matrix, axis=1)),
                            np.argmax(np.max(iou_matrix, axis=0))]
                iou_matrix[max_args[0]] = 0
                matched_pair.append(max_args)
            matched_pair_list.append(matched_pair)
        return matched_pair_list
    
    def _calc_conf(self, batch_labels, matched_pair_list, surrounding_pair_list):
        batch_conf = np.zeros([len(batch_labels),
                               self._default_box.shape[0],
                               self._num_classes])
        batch_conf[:, :, 0] = 1.0
        for i, labels in enumerate(batch_labels, 0):
            for pair in matched_pair_list[i]:
                batch_conf[i, pair[0]] = labels[pair[1]]
            for pair in surrounding_pair_list[i]:
                batch_conf[i, pair[0]] = labels[pair[1]]
        return batch_conf

    def _calc_loc(self, batch_bboxes, matched_pair_list):
        batch_loc = np.zeros([len(batch_bboxes),
                              self._default_box.shape[0],
                              4])
        for i, bboxes in enumerate(batch_bboxes, 0):
            for pair in matched_pair_list[i]:
                d_cx, d_cy, d_w, d_h = self._default_box[pair[0]]
                b_xmin, b_ymin, b_xmax, b_ymax = bboxes[pair[1]]
                b_cx, b_cy = (b_xmax + b_xmin) / 2, (b_ymax + b_ymin) / 2
                b_w, b_h = b_xmax - b_xmin, b_ymax - b_ymin
                delta_cx = (b_cx - d_cx) / (d_w * 0.1)
                delta_cy = (b_cy - d_cy) / (d_h * 0.1)
                delta_w = np.log(b_w / d_w) / 0.2
                delta_h = np.log(b_h / d_h) / 0.2
                batch_loc[i, pair[0]] = [delta_cx, delta_cy, delta_w, delta_h]
        return batch_loc

    def _anchor_box_encode(self, batch_labels, batch_bboxes):
        iou_matrix_list = self._calc_iou_matrix(batch_bboxes)
        matched_pair_list =  self._calc_matched_pair(iou_matrix_list)
        iou_matrix_list = self._remove_matched_iou(iou_matrix_list,
                                                   matched_pair_list)
        surrounding_pair_list = self._calc_surrounding_pair(iou_matrix_list)
        batch_conf = self._calc_conf(batch_labels,
                                     matched_pair_list,
                                     surrounding_pair_list)
        batch_loc = self._calc_loc(batch_bboxes, matched_pair_list)
        print(batch_loc)
        return batch_conf, batch_loc

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
            batch_images, batch_labels, batch_bboxes = self._load_batch_data(batch_data)
            batch_images, batch_labels, batch_bboxes = self._augment(
                                                         batch_images,
                                                         batch_labels,
                                                         batch_bboxes
                                                     )
            batch_conf, batch_loc = self._anchor_box_encode(batch_labels,
                                                            batch_bboxes)
            batch_images = np.transpose(batch_images, (0, 3, 1, 2))
            queue.put({'data_id'   : [data['data_id'] for data in batch_data],
                       'image'     : np.array(batch_images).astype(np.float32),
                       'conf'      : np.array(batch_conf).astype(np.float32),
                       'loc'       : np.array(batch_loc).astype(np.float32),
                       'label'     : batch_labels,
                       'bbox'      : batch_bboxes,
                       'worker_id' : worker_id,
                       'is_end'    : False if i < num_batches - 1 else True})
    
