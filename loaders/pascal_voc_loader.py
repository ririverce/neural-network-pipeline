import os
import csv
import json
import pickle
import random

import tqdm



class PascalVOCLoader:

    train_label_file = 'train.json'
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(os.path.join(self.root_dir, self.train_label_file), 'r') as f:
            dataset = json.load(f)
            for i in range(len(dataset)):
                dataset[i]['data_id'] = dataset[i]['image_path'].split('/')[-1]
                dataset[i]['image_path'] = os.path.join(self.root_dir,
                                                        dataset[i]['image_path'])
            random.shuffle(dataset)
            self.train = dataset[:int(len(dataset)*0.9)] 
            self.valid = dataset[int(len(dataset)*0.9):]

    def load(self):
        return self.train, self.valid