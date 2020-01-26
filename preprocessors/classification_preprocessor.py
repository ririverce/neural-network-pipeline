import random

import numpy as np
import cv2



class ClassificationPreprocessor:

    def __init__(self, dataset, batch_size,
                 output_size=(64, 64), is_train=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.output_size = output_size
        self.is_train = is_train

    def __random_crop_and_resize(self, image):
        image_height, image_width = image.shape[:2]
        crop_x_min = image_width * random.random() * 0.1
        crop_y_min = image_height * random.random() * 0.1
        crop_x_max = image_width * (random.random() * 0.1 + 0.9)
        crop_y_max = image_height * (random.random() * 0.1 + 0.9)
        image = image[int(crop_y_min):int(crop_y_max),
                      int(crop_x_min):int(crop_x_max)]
        image = cv2.resize(image, self.output_size)
        return image

    def __random_rotate(self, image):
        image = 255 - image
        image_height, image_width = image.shape[:2]
        center = (int(image_width/2), int(image_width/2))
        angle = random.random() * 360
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (image_width, image_height))
        image = 255 - image
        return image        

    def __random_cutout(self, image, size_ratio=0.2, num_cut=3):
        image_height, image_width = image.shape[:2]
        for i in range(num_cut):
            crop_x_min = image_width * (1 - size_ratio) * random.random()
            crop_y_min = image_height * (1 - size_ratio) * random.random()
            crop_x_max = image_width * size_ratio + crop_x_min
            crop_y_max = image_height * size_ratio + crop_y_min
            color = random.random() * 255
            image[int(crop_y_min):int(crop_y_max), int(crop_x_min):int(crop_x_max)] = int(color)
        return image

    def __random_brightness(self, image, rand_range=(0.8, 1.2)):
        scale = random.random() * (rand_range[1] - rand_range[0]) + rand_range[0]
        image = image * scale
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image

    def __add_gaussian_nois(self, image, mean=0, max_var=10):
        var = random.random() * max_var
        gauss = np.random.normal(mean, var, image.shape)
        image = image + gauss
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def run(self, queue):
        random.shuffle(self.dataset)
        batch_image = []
        batch_label = []
        meta = {"skip" : 0, "end" : False}
        for data in self.dataset:
            image = cv2.imread(data["image_path"])
            if image is None:
                meta["skip"] += 1
                continue
            if "grapheme_root" in data.keys():
                label = [data["grapheme_root"],
                         data["vowel_diacritic"],
                         data["consonant_diacritic"]]
            else:
                label = data["label"]
            if self.is_train:
                image = self.__random_crop_and_resize(image)
                image = self.__random_rotate(image)
                image = self.__random_cutout(image)
                image = self.__random_brightness(image)
                image = self.__add_gaussian_nois(image)
                #image = cv2.resize(image, self.output_size)
            else:
                image = cv2.resize(image, self.output_size)
            #cv2.imshow("test", image)
            #cv2.waitKey(1000)
            #image = image[:, :, 0]
            #image = np.expand_dims(image, axis=-1)
            image = np.transpose(image, (2, 0, 1))
            image = (image - 127.5) / 127.5
            batch_image.append(image)
            batch_label.append(label)
            if len(batch_image) == self.batch_size:
                queue.put([np.array(batch_image).astype(np.float32),
                           np.array(batch_label).astype(np.int),
                           meta])
                batch_image = []
                batch_label = []
                meta = {"skip" : 0, "end" : False}
        meta["skip"] += len(batch_image)    
        meta["end"] = True
        queue.put([np.array(batch_image).astype(np.float32),
                   np.array(batch_label).astype(np.int),
                   meta])        



def test():
    import os
    import sys
    sys.path.append("../")
    import multiprocessing
    import cv2
    from loaders import bengaliai_loader
    root_dir = "../dataset/bengaliai-cv19/"
    loader = bengaliai_loader.BengaliAILoader(root_dir)
    dataset = loader.load()
    preprocessor = ClassificationPreprocessor(dataset, 64)
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=preprocessor.run,
                                      args=(queue,))
    process.start()
    while True:
        batch_image, batch_label, meta = queue.get()
        if meta["end"]:
            break
        for image in batch_image:
            cv2.imshow("test", image)
            cv2.waitKey(1)



if __name__ == "__main__":
    test()