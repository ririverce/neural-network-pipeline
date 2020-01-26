import random
import time

import tqdm
import numpy as np
import cv2
import torch

import loaders
import preprocessors
import iterators
import models



def main():

    device = "cuda:0"

    root_dir = "./dataset/CIFAR-10/"
    loader = loaders.cifar10_loader.Cifar10Loader(root_dir)
    dataset = loader.load()
    random.seed(a=0)
    random.shuffle(dataset)
    preprocessor_constructor = preprocessors.classification_preprocessor.ClassificationPreprocessor
    train_iterator = iterators.classification_iterator.ClassificationIterator(
                         dataset,
                         preprocessor_constructor,
                         image_size=(32, 32),
                         batch_size=512,
                         num_processes=4,
                         phase="train")
    validation_iterator = iterators.classification_iterator.ClassificationIterator(
                              dataset,
                              preprocessor_constructor,
                              image_size=(32, 32),
                              batch_size=512,
                              num_processes=4,
                              phase="validation")
    net = models.alexnet.AlexNet(11).to(device)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(1000):

        print(f"[epoch {epoch:04d}]")

        """ train """
        total_skip = 0
        loss = 0
        correct = 0
        pbar = tqdm.tqdm(total=train_iterator.num_samples)
        pbar.set_description("train ")
        for batch in train_iterator:
            # initialize
            optimizer.zero_grad()
            # input
            batch_image = torch.from_numpy(batch[0]).to(device)
            # infer
            outputs = net(batch_image)
            # to gpu
            label = torch.from_numpy(batch[1]).to(device)
            # calc loss
            batch_loss = loss_function(outputs, label)
            # backward
            batch_loss.backward()
            # optimize
            optimizer.step()
            # accumulate loss
            loss += batch_loss.item() * len(batch_image)
            # accumulate correct
            outputs = outputs.cpu().detach().numpy()
            infer = np.argmax(outputs, axis=-1)
            label = label.cpu().detach().numpy()
            batch_correct = np.equal(infer, label)
            correct += np.count_nonzero(batch_correct)
            # post process
            total_skip += batch[2]["skip"]
            pbar.update(len(batch_image)+batch[2]["skip"])
        pbar.close()
        loss /= train_iterator.num_samples - total_skip
        correct /= train_iterator.num_samples - total_skip
        print(f"result : loss={loss:.4f}, accuracy={correct:.4f}")

        with torch.no_grad():
            """ validation """
            total_skip = 0
            loss = 0
            correct = 0
            pbar = tqdm.tqdm(total=validation_iterator.num_samples)
            pbar.set_description("validation ")
            for batch in validation_iterator:
                # initialize
                optimizer.zero_grad()
                # input
                batch_image = torch.from_numpy(batch[0]).to(device)
                # infer
                outputs = net(batch_image)
                # to gpu
                label = torch.from_numpy(batch[1]).to(device)
                # calc loss
                batch_loss = loss_function(outputs, label)
                # accumulate loss
                loss += batch_loss.item() * len(batch_image)
                # accumulate correct
                outputs = outputs.cpu().detach().numpy()
                infer = np.argmax(outputs, axis=-1)
                label = label.cpu().detach().numpy()
                batch_correct = np.equal(infer, label)
                correct += np.count_nonzero(batch_correct)
                # post process
                total_skip += batch[2]["skip"]
                pbar.update(len(batch_image)+batch[2]["skip"])
            pbar.close()
            loss /= validation_iterator.num_samples - total_skip
            correct /= validation_iterator.num_samples - total_skip
            print(f"result : loss={loss:.4f}, accuracy={correct:.4f}")

        if epoch % 10 == 9:
            torch.save(net.state_dict(), f"./result/epoch_{epoch:04d}.model")



if __name__ == "__main__":
    main()