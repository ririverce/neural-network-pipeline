import csv

import tqdm
import numpy as np
import cv2
import torch

import loaders
import processors
import iterators
import models
import losses
import loggers



def main():

    batch_size = 256
    num_classes = 1000
    image_size = (128, 128)

    """ load dataset """
    dataset = loaders.ImageNetLoader('./datasets/ImageNet').load()
    train_dataset, valid_dataset = dataset

    """ processor """
    train_processor = processors.ImageNetClassificationProcessor(
                          batch_size,
                          num_classes=num_classes,
                          enable_augmentation=True,
                          image_size=image_size
                      )
    valid_processor = processors.ImageNetClassificationProcessor(
                          batch_size,
                          num_classes=num_classes,
                          enable_augmentation=False,
                          image_size=image_size
                      )

    """ iterator """
    train_iterator = iterators.MultiprocessIterator(train_dataset,
                                                    train_processor,
                                                    num_workers=4)
    valid_iterator = iterators.MultiprocessIterator(valid_dataset,
                                                    valid_processor,
                                                    num_workers=4)

    """ device """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    """ model """
    model = models.ResNet18(input_channels=3,
                            num_classes=num_classes).to(device)

    """ loss """
    loss_function = losses.CrossEntropyLoss().to(device)

    """ optimizer """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=20)

    """ logger """
    logger = loggers.SimpleLogger()

    """ learning """    
    for epoch in range(10):
        print(f"-"*64)
        print(f"[epoch {epoch:>4d}]")
        phase = 'train'
        torch.set_grad_enabled(True)
        for batch_data in tqdm.tqdm(train_iterator, desc=phase):
            optimizer.zero_grad()
            batch_image = torch.from_numpy(batch_data['image']).to(device)
            batch_target = torch.from_numpy(batch_data['target']).to(device)
            batch_output = model(batch_image)
            batch_loss = loss_function(batch_output, batch_target)
            batch_loss.sum().backward()
            optimizer.step()
            batch_loss = batch_loss.data.cpu().numpy()
            batch_label = np.argmax(batch_target.data.cpu().numpy(), axis=-1).flatten()
            batch_pred = np.argmax(batch_output.data.cpu().numpy(), axis=-1).flatten()
            logger.add_batch_loss(batch_loss, phase=phase)
            logger.add_batch_pred(batch_pred, phase=phase)
            logger.add_batch_label(batch_label, phase=phase)
        loss = logger.get_loss(phase)
        accuracy = logger.get_accuracy(phase)
        print(f"loss : {loss}")
        print(f"accuracy : {accuracy}")
        phase = 'valid'
        torch.set_grad_enabled(False)
        for batch_data in tqdm.tqdm(valid_iterator, desc=phase):
            optimizer.zero_grad()
            batch_image = torch.from_numpy(batch_data['image']).to(device)
            batch_target = torch.from_numpy(batch_data['target']).to(device)
            batch_output = model(batch_image)
            batch_loss = loss_function(batch_output, batch_target)
            batch_loss = batch_loss.data.cpu().numpy()
            batch_label = np.argmax(batch_target.data.cpu().numpy(), axis=-1).flatten()
            batch_pred = np.argmax(batch_output.data.cpu().numpy(), axis=-1).flatten()
            logger.add_batch_loss(batch_loss, phase=phase)
            logger.add_batch_pred(batch_pred, phase=phase)
            logger.add_batch_label(batch_label, phase=phase)
        loss = logger.get_loss(phase)
        accuracy = logger.get_accuracy(phase)
        print(f"loss : {loss:.4f}")
        print(f"accuracy : {accuracy:.4f}")
        logger.step()



if __name__ == "__main__":
    main()