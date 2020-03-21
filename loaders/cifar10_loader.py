import os
import csv
import json
import pickle
import random

import tqdm



class Cifar10Loader:

    train_label_file = 'train.json'
    test_label_file = 'test.json'
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, self.train_label_file), 'r') as f:
            dataset = json.load(f)
            for i in range(len(dataset)):
                dataset[i]['data_id'] = dataset[i]['image_path']
                dataset[i]['image_path'] = os.path.join(self.root_dir,
                                                        dataset[i]['image_path'])
            self.train = dataset[:int(len(dataset)*0.8)] 
            self.valid = dataset[int(len(dataset)*0.8):]
        with open(os.path.join(self.root_dir, self.test_label_file), 'r') as f:
            self.test = json.load(f)

    def load(self):
        return self.train, self.valid