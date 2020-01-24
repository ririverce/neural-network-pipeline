import multiprocessing



class ClassificationIterator:

    def __init__(self, dataset, preprocessor_constructor,
                 image_size=(64, 64), batch_size=64,
                 num_processes=2, phase="train"):
        self.dataset = dataset
        self.preprocessor_constructor = preprocessor_constructor
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.phase = phase
        if phase == "train":
            self.dataset = self.dataset[:int(len(self.dataset)*0.8)]
        elif phase == "validation":
            self.dataset = self.dataset[int(len(self.dataset)*0.8):]
        self.num_samples = len(self.dataset)
        self.split_dataset = [self.dataset[i::self.num_processes] for i in range(self.num_processes)]
        self.batch_queue = multiprocessing.Queue()
        self.end_count = 0
        self.initialized = False
    
    def __iter__(self):
        return self

    def __next__(self):
        if not self.initialized:
            self.process_list = []
            for d in self.split_dataset:
                preprocessor = self.preprocessor_constructor(d, self.batch_size,
                                                             output_size=self.image_size)
                self.process_list.append(multiprocessing.Process(target=preprocessor.run,
                                                                 args=(self.batch_queue,)))
            for p in self.process_list:
                    p.start()
            self.initialized = True
        batch = self.batch_queue.get()
        if batch[2]["end"]:
            self.end_count += 1
            if self.end_count >= self.num_processes:
                self.end_count = 0
                for p in self.process_list:
                    p.join()
                self.initialized = False
                raise StopIteration()
        return batch



def test():
    import sys
    sys.path.append("../")
    import cv2
    from loaders import bengaliai_loader
    from preprocessors import classification_preprocessor

    root_dir = "../dataset/bengaliai-cv19/"
    loader = bengaliai_loader.BengaliAILoader(root_dir)
    dataset = loader.load()
    preprocessor_constructor = classification_preprocessor.ClassificationPreprocessor
    iterator = ClassificationIterator(dataset,
                                      preprocessor_constructor,
                                      image_size=(64, 64),
                                      batch_size=64,
                                      num_processes=16,
                                      phase="train")
    count = 0
    for batch in iterator:
        for image in batch[0]:
            #cv2.imshow("test", image)
            #cv2.waitKey(1)
            count += 1
    print(f"count = {count}")



if __name__ == "__main__":
    test()