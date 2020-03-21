import os
import csv
import json
import pickle
import random

import tqdm



class ImageNetLoader:

    train_label_file = 'train.json'
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, self.train_label_file), 'r') as f:
            self.dataset = json.load(f)
            for i in range(len(self.dataset)):
                self.dataset[i]['data_id'] = self.dataset[i]['image_path']
                self.dataset[i]['image_path'] = os.path.join(
                                                    os.path.join(self.root_dir,
                                                                 'train'),
                                                    self.dataset[i]['image_path']
                                                )

    def load(self):
        random.shuffle(self.dataset)
        num_data = len(self.dataset)
        self.train = self.dataset[:int(num_data*0.8)]
        self.valid = self.dataset[int(num_data*0.8):]
        return self.train, self.valid