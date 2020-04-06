import csv

import tqdm
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import loaders
import processors
import iterators
import models
import losses
import loggers
import utils



def main():

    batch_size = 16
    num_classes = 20
    image_size = (256, 256)

    """ load dataset """
    dataset = loaders.BDD100KSegmentationLoader('./datasets/BDD100K').load()
    train_dataset, valid_dataset = dataset

    """ processor """
    train_processor = processors.BDD100KSegmentationProcessor(
                          batch_size,
                          num_classes=num_classes,
                          image_size=image_size,
                          enable_augmentation=True,
                      )
    valid_processor = processors.BDD100KSegmentationProcessor(
                          batch_size,
                          num_classes=num_classes,
                          image_size=image_size,
                          enable_augmentation=False,
                      )

    """ iterator """
    train_iterator = iterators.MultiprocessIterator(train_dataset,
                                                    train_processor,
                                                    num_workers=1)
    valid_iterator = iterators.MultiprocessIterator(valid_dataset,
                                                    valid_processor,
                                                    num_workers=2)

    """ device """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    """ model """
    model = models.LiteUNet(input_channels=3,
                            num_classes=num_classes).to(device)

    """ optimizer """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                       T_max=20)

    """ logger """
    logger = loggers.SimpleLogger()

    """ learning """    
    for epoch in range(1, 200):
        print(f"-"*64)
        print(f"[epoch {epoch:>4d}]")
        phase = 'train'
        torch.set_grad_enabled(True)
        i = 0
        for batch_data in tqdm.tqdm(train_iterator, desc=phase):
            optimizer.zero_grad()
            batch_image = torch.from_numpy(batch_data['image']).to(device)
            batch_segment = torch.from_numpy(batch_data['segment']).to(device)
            batch_output = model(batch_image)
            batch_loss = -1 * batch_segment * F.log_softmax(batch_output, 1)
            batch_loss = (1 - F.softmax(batch_output, 1)).pow(2) * batch_loss
            batch_loss = batch_loss.sum(1).mean(1).mean(1)
            batch_loss.mean().backward()
            optimizer.step()
            batch_loss = batch_loss.detach().cpu().numpy()
            logger.add_batch_loss(batch_loss, phase=phase)
            if i >= 32:
                continue
            batch_image = batch_image.detach().cpu().numpy()
            batch_output = F.softmax(batch_output, 1).detach().cpu().numpy()
            batch_segment = batch_segment.detach().cpu().numpy()
            for image, output, segment in zip(batch_image, batch_output, batch_segment):
                image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
                output = utils.segment_decode.voc_color_decode(output)
                segment = utils.segment_decode.voc_color_decode(segment)
                bg = np.zeros([image.shape[0]*2, image.shape[1]*2, 3])
                bg[:image.shape[0], :image.shape[1]] = image
                bg[:image.shape[0], image.shape[1]:] = segment
                bg[image.shape[0]:, :image.shape[1]] = (output * 0.5 + image * 0.5).astype(np.uint8)
                bg[image.shape[0]:, image.shape[1]:] = output
                image = bg
                cv2.imwrite(f'./results/train_images/{epoch:04d}_{i:08d}.jpg', image)
                i += 1
                break
        loss = logger.get_loss(phase)
        print(f"loss : {loss:.4f}")
        phase = 'valid'
        torch.set_grad_enabled(False)
        i = 0
        for batch_data in tqdm.tqdm(valid_iterator, desc=phase):
            optimizer.zero_grad()
            batch_image = torch.from_numpy(batch_data['image']).to(device)
            batch_segment = torch.from_numpy(batch_data['segment']).to(device)
            batch_output = model(batch_image)
            batch_loss = -1 * batch_segment * F.log_softmax(batch_output, 1)
            batch_loss = (1 - F.softmax(batch_output, 1)).pow(2) * batch_loss
            batch_loss = batch_loss.sum(1).mean(1).mean(1)
            batch_loss = batch_loss.detach().cpu().numpy()
            logger.add_batch_loss(batch_loss, phase=phase)
            if i >= 32:
                continue
            batch_image = batch_image.detach().cpu().numpy()
            batch_output = F.softmax(batch_output, 1).detach().cpu().numpy()
            batch_segment = batch_segment.detach().cpu().numpy()
            for image, output, segment in zip(batch_image, batch_output, batch_segment):
                image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
                output = utils.segment_decode.voc_color_decode(output)
                segment = utils.segment_decode.voc_color_decode(segment)
                bg = np.zeros([image.shape[0]*2, image.shape[1]*2, 3])
                bg[:image.shape[0], :image.shape[1]] = image
                bg[:image.shape[0], image.shape[1]:] = segment
                bg[image.shape[0]:, :image.shape[1]] = (output * 0.5 + image * 0.5).astype(np.uint8)
                bg[image.shape[0]:, image.shape[1]:] = output
                image = bg
                cv2.imwrite(f'./results/valid_images/{epoch:04d}_{i:08d}.jpg', image)
                i += 1
                break
        loss = logger.get_loss(phase)
        print(f"loss : {loss:.4f}")
        logger.step()

        #scheduler.step()

        torch.save(model.state_dict(), f"./results/model/epoch_{epoch:04d}.model")



if __name__ == "__main__":
    main()