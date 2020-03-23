import copy

import numpy as np



class SimpleLogger:

    def __init__(self):
        self._metrics = {'train' : {'loss'  : [],
                                    'pred'  : [],
                                    'label' : []},
                         'valid' : {'loss'  : [],
                                    'pred'  : [],
                                    'label' : []}}
        self._log = [] 

    def add_batch_loss(self, batch_loss, phase=None):
        self._metrics[phase]['loss'].extend(batch_loss)

    def add_batch_pred(self, batch_pred, phase=None):
        self._metrics[phase]['pred'].extend(batch_pred)

    def add_batch_label(self, batch_label, phase=None):
        self._metrics[phase]['label'].extend(batch_label)

    def get_loss(self, phase):
        loss = self._metrics[phase]['loss']
        return np.mean(loss)
        
    def get_accuracy(self, phase):
        pred = self._metrics[phase]['pred']
        label = self._metrics[phase]['label']
        return np.count_nonzero(np.equal(pred, label)) / len(pred)

    def step(self):
        self._log.append(copy.deepcopy(self._metrics))
        self._metrics = {'train' : {'loss'  : [],
                                    'pred'  : [],
                                    'label' : []},
                         'valid' : {'loss'  : [],
                                    'pred'  : [],
                                    'label' : []}}        