import os
import csv
import json
import pickle
import random

import tqdm



class BDD100KLoader:

    train_label_file = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    valid_label_file = 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    train_image_sub_dir = 'bdd100k_images/bdd100k/images/100k/train'
    valid_image_sub_dir = 'bdd100k_images/bdd100k/images/100k/val'
    label_num_map = {'car' : 1,
                     'truck' : 2,
                     'bus' : 3,
                     'motor' : 4,
                     'bike' : 5,
                     'person' : 6,
                     'rider' : 7,
                     'traffic sign' : 8,
                     'traffic light' : 9,
                     'train' : 10}
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, self.train_label_file), 'r') as f:
            dataset = json.load(f)
        image_dir = os.path.join(self.root_dir, self.train_image_sub_dir)
        processed_dataset = []
        for i in range(len(dataset)):
            data_id = dataset[i]['name']
            image_path = os.path.join(image_dir, dataset[i]['name'])
            labels = []
            bboxes = []
            for l in dataset[i]['labels']:
                if 'box2d' in l.keys():
                    labels.append(self.label_num_map[l['category']])
                    x_min = l['box2d']['x1'] / 1280
                    y_min = l['box2d']['y1'] / 720
                    x_max = l['box2d']['x2'] / 1280
                    y_max = l['box2d']['y2'] / 720
                    bboxes.append([x_min, y_min, x_max, y_max])
            processed_dataset.append({'data_id' : data_id,
                                      'image_path' : image_path,
                                      'label' : labels,
                                      'bbox' : bboxes})
        del dataset
        dataset = processed_dataset
        self.train = dataset[:int(len(dataset)*0.8)] 
        self.valid = dataset[int(len(dataset)*0.8):]

    def load(self):
        return self.train, self.valid



def test():
    root_dir = 'datasets/BDD100K'
    loader = BDD100KLoader(root_dir)
    


if __name__ == '__main__':
    test()