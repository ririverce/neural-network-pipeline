import random
import multiprocessing



class MultiprocessIterator:

    def __init__(self, dataset, preprocessor,
                 num_workers=1, enable_shuffle=True):
        self._dataset = dataset
        self._preprocessor = preprocessor
        self._num_workers = num_workers
        self._enable_shuffle = enable_shuffle
        self._worker_list = []
        self._data_queue = multiprocessing.Queue()
        self._initialized = False
        
    def _initialize_workers(self):
        if self._enable_shuffle:
            random.shuffle(self._dataset)
        self._devided_dataset = [self._dataset[i::self._num_workers] \
                                 for i in range(self._num_workers)]
        self._worker_list = []
        for worker_id, worker_data in enumerate(self._devided_dataset, 0):
            self._worker_list.append(
                multiprocessing.Process(
                    target=self._preprocessor,
                    args=(worker_data, self._data_queue, worker_id,)
                )
            )
        self._terminated_worker_list = []
        for worker in self._worker_list:
            worker.start()
        self._initialized = True

    def __len__(self):
        """ FIXME """
        batch_size = self._preprocessor._batch_size
        total_data_size = len(self._dataset)
        return total_data_size // batch_size + 1

    def __iter__(self):
        return self

    def __next__(self):
        if not self._initialized:
            self._initialize_workers()
        if len(self._terminated_worker_list) == self._num_workers:
            self._initialized = False
            raise StopIteration()
        data = self._data_queue.get()
        if data['is_end']:
            worker_id = data['worker_id']
            self._terminated_worker_list.append(worker_id)
        return data



"""**********************
********** TEST *********
**********************"""
def test():
    """ settings """
    dataset = [{'value' : random.random()} for _ in range(10000)]
    class Preprocess:
        def __call__(self, dataset, queue, worker_id):
            batch_size = 128
            batch = {'value' : [],
                     'end_flag' : False,
                     'worker_id' : worker_id}
            for data in dataset:
                batch['value'].append(data['value']**2)
                if len(batch) == batch_size:
                    queue.put(batch)
                    batch = {'value' : [],
                             'is_end' : False}
            batch['is_end'] = True
            queue.put(batch)
    """ test """
    iterator = MultiprocessIterator(dataset, Preprocess(), 128, num_workers=4)
    processed_data = []
    for epoch in range(100):
        processed_data = []
        for batch in iterator:
            processed_data.extend(batch['value'])
        print(f"epoch {epoch:04d} : processed {len(processed_data)}")
    


if __name__ == '__main__':
    test()