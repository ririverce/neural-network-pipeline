import time
import random

import numpy as np
import cv2



class Word2VecProcessor:

    def __init__(self, batch_size, words, num_neighbors):
        self._batch_size = batch_size
        self._words = words
        self._num_classes = len(words)
        self._num_neighbors = num_neighbors

    def __call__(self, dataset, queue, worker_id):
        batch_input = []
        batch_target = []
        for data in dataset:
            split_data = data.split(' ')
            int_split_data = []
            for w in split_data:
                if w == '\n':
                    int_split_data.append(0)
                elif w in self._words.keys():
                    int_split_data.append(self._words[w])
            for i, label in enumerate(int_split_data, 0):
                if i < self._num_neighbors or \
                   i + self._num_neighbors >= len(int_split_data):
                     continue
                batch_input.append(label)
                neighbor_words = []
                for j in range(self._num_neighbors):
                    neighbor_words.append(int_split_data[i - j])
                for j in range(self._num_neighbors):
                    neighbor_words.append(int_split_data[i + j])
                batch_target.append(neighbor_words)
                if len(batch_input) >= self._batch_size:
                    batch_input = np.array(batch_input, dtype=np.long)
                    batch_target = np.array(batch_target, dtype=np.long)
                    while queue.qsize() >= 1024:
                        time.sleep(0.001)
                    queue.put({'input' : batch_input,
                               'target' : batch_target,
                               'worker_id' : worker_id,
                               'is_end' : False})
                    batch_input = []
                    batch_target = []
        batch_input = np.array(batch_input, dtype=np.int32)
        batch_target = np.array(batch_target, dtype=np.int32)
        queue.put({'input' : batch_input,
                   'target' : batch_target,
                   'worker_id' : worker_id,
                   'is_end' : True})
    

    
def unit_test():
    import multiprocessing
    import sys
    sys.path.append('../')
    import loaders
    loader = loaders.WikipediaJPLoader('../datasets/WikipediaJP')
    documents, words = loader.load()
    processor = Word2VecProcessor(128, words, 5)
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=processor, 
                                      args=[documents, queue, 0])
    process.start()
    while True: 
        data = queue.get()
        print(data['input'].shape, data['target'].shape)
        if data['is_end']:
            break


if __name__ == '__main__':
    unit_test()