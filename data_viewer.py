import csv

import tqdm
import numpy as np
import cv2

import loaders
import processors
import iterators



def main():

    batch_size = 128
    image_size = (128, 128)

    """ load dataset """
    dataset = loaders.ImageNetLoader('./datasets/ImageNet').load()
    train_dataset, valid_dataset = dataset

    """ processor """
    train_processor = processors.ImageNetClassificationProcessor(
                          batch_size,
                          num_classes=1000,
                          enable_augmentation=True,
                          image_size=image_size
                      )

    """ iterator """
    train_iterator = iterators.MultiprocessIterator(train_dataset,
                                                    train_processor,
                                                    num_workers=1)
    
    for batch_data in tqdm.tqdm(train_iterator):
        batch_image = np.transpose(batch_data['image'], (0, 2, 3, 1))
        for image in batch_image:
            #image = image * 127.5 + 127.5
            #image = image * 255.0
            image = image.astype(np.uint8)
            #image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            cv2.imshow("test", image)
            cv2.waitKey(100)



if __name__ == "__main__":
    main()