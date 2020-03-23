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
import utils



def main():

    batch_size = 48
    num_classes = 21
    image_size = (300, 300)

    """ load dataset """
    dataset = loaders.PascalVOCLoader('./datasets/PascalVOC').load()
    train_dataset, valid_dataset = dataset

    """ default box """
    default_box = utils.anchor_box_utils.generate_default_box(
                      image_size,
                      utils.anchor_box_utils.num_grids_ssd300,
                      utils.anchor_box_utils.grid_step_ssd300,
                      utils.anchor_box_utils.grid_size_ssd300,
                      utils.anchor_box_utils.aspect_ratio_ssd300
                  )

    """ processor """
    train_processor = processors.AnchorBoxProcessor(
                          batch_size,
                          num_classes=num_classes,
                          default_box=default_box,
                          image_size=image_size,
                          enable_augmentation=True,
                      )
    valid_processor = processors.AnchorBoxProcessor(
                          batch_size,
                          num_classes=num_classes,
                          default_box=default_box,
                          image_size=image_size,
                          enable_augmentation=False,
                      )

    """ iterator """
    train_iterator = iterators.MultiprocessIterator(train_dataset,
                                                    train_processor,
                                                    num_workers=2)
    valid_iterator = iterators.MultiprocessIterator(valid_dataset,
                                                    valid_processor,
                                                    num_workers=2)

    """ device """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    """ model """
    model = models.SSD300LiteVGG16(input_channels=3,
                                   num_classes=num_classes,
                                   num_bboxes=[4, 6, 6, 6, 4, 4]).to(device)

    """ loss """
    loss_function = losses.MultiBoxLoss().to(device)

    """ optimizer """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=20)

    """ logger """
    logger = loggers.SimpleLogger()

    """ learning """    
    for epoch in range(100):
        print(f"-"*64)
        print(f"[epoch {epoch:>4d}]")
        phase = 'train'
        torch.set_grad_enabled(True)
        for batch_data in tqdm.tqdm(train_iterator, desc=phase):
            optimizer.zero_grad()
            batch_image = torch.from_numpy(batch_data['image']).to(device)
            batch_gt_conf = torch.from_numpy(batch_data['conf']).to(device)
            batch_gt_loc = torch.from_numpy(batch_data['loc']).to(device)
            batch_output = model(batch_image)
            batch_pred_conf, batch_pred_loc = batch_output
            batch_loss_conf, batch_loss_loc = loss_function(batch_pred_conf,
                                                            batch_pred_loc,
                                                            batch_gt_conf,
                                                            batch_gt_loc)
            batch_loss = batch_loss_conf + batch_loss_loc
            batch_loss.sum().backward()
            optimizer.step()
            batch_loss = batch_loss.detach().cpu().numpy()
            logger.add_batch_loss(batch_loss, phase=phase)
        loss = logger.get_loss(phase)
        print(f"loss : {loss}")
        phase = 'valid'
        torch.set_grad_enabled(False)
        for batch_data in tqdm.tqdm(valid_iterator, desc=phase):
            optimizer.zero_grad()
            batch_image = torch.from_numpy(batch_data['image']).to(device)
            batch_gt_conf = torch.from_numpy(batch_data['conf']).to(device)
            batch_gt_loc = torch.from_numpy(batch_data['loc']).to(device)
            batch_output = model(batch_image)
            batch_pred_conf, batch_pred_loc = batch_output
            batch_loss_conf, batch_loss_loc = loss_function(batch_pred_conf,
                                                            batch_pred_loc,
                                                            batch_gt_conf,
                                                            batch_gt_loc)
            batch_loss = batch_loss_conf + batch_loss_loc
            batch_loss = batch_loss.data.cpu().numpy()
            logger.add_batch_loss(batch_loss, phase=phase)
        loss = logger.get_loss(phase)
        print(f"loss : {loss:.4f}")
        logger.step()



if __name__ == "__main__":
    main()