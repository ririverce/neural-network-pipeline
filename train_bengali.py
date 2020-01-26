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

    root_dir = "./dataset/bengaliai-cv19/"
    loader = loaders.bengaliai_loader.BengaliAILoader(root_dir)
    dataset = loader.load()
    random.seed(a=0)
    random.shuffle(dataset)
    preprocessor_constructor = preprocessors.classification_preprocessor.ClassificationPreprocessor
    train_iterator = iterators.classification_iterator.ClassificationIterator(
                         dataset,
                         preprocessor_constructor,
                         image_size=(128, 128),
                         batch_size=32,
                         num_processes=2,
                         phase="train")
    validation_iterator = iterators.classification_iterator.ClassificationIterator(
                              dataset,
                              preprocessor_constructor,
                              image_size=(128, 128),
                              batch_size=32,
                              num_processes=2,
                              phase="validation")
    net = models.vgg16.BengaliVGG16([168, 11, 7]).to(device)
    loss_function_graph = torch.nn.CrossEntropyLoss().to(device)
    loss_function_vowel = torch.nn.CrossEntropyLoss().to(device)
    loss_function_conso = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(1000):

        print(f"[epoch {epoch:04d}]")

        """ train """
        print("train")
        total_skip = 0
        loss_graph = 0
        loss_vowel = 0
        loss_conso = 0
        loss = 0
        accuracy_graph = 0
        accuracy_vowel = 0
        accuracy_conso = 0
        pbar = tqdm.tqdm(total=train_iterator.num_samples, leave=True, unit="samples")
        for batch in train_iterator:
            # initialize
            optimizer.zero_grad()
            # input
            batch_image = torch.from_numpy(batch[0]).to(device)
            # infer
            outputs_graph, outputs_vowel, outputs_conso = net(batch_image)
            # label
            label_graph = torch.from_numpy(batch[1][:, 0]).to(device)
            label_vowel = torch.from_numpy(batch[1][:, 1]).to(device)
            label_conso = torch.from_numpy(batch[1][:, 2]).to(device)
            # calc loss
            batch_loss_graph = loss_function_graph(outputs_graph, label_graph)
            batch_loss_vowel = loss_function_vowel(outputs_vowel, label_vowel)
            batch_loss_conso = loss_function_conso(outputs_conso, label_conso)
            batch_loss = batch_loss_graph + batch_loss_vowel + batch_loss_conso
            # backward
            batch_loss.backward()
            # optimize
            optimizer.step()
            # accumulate loss
            loss_graph += batch_loss_graph.item() * len(batch_image)
            loss_vowel += batch_loss_vowel.item() * len(batch_image)
            loss_conso += batch_loss_conso.item() * len(batch_image)
            loss += batch_loss.item() * len(batch_image)
            # infer result
            infer_graph = np.argmax(outputs_graph.cpu().detach().numpy(), axis=-1)
            infer_vowel = np.argmax(outputs_vowel.cpu().detach().numpy(), axis=-1)
            infer_conso = np.argmax(outputs_conso.cpu().detach().numpy(), axis=-1)
            # label
            label_graph = label_graph.cpu().detach().numpy()
            label_vowel = label_vowel.cpu().detach().numpy()
            label_conso = label_conso.cpu().detach().numpy()
            # score
            accuracy_graph += np.count_nonzero(np.equal(infer_graph, label_graph))
            accuracy_vowel += np.count_nonzero(np.equal(infer_vowel, label_vowel))
            accuracy_conso += np.count_nonzero(np.equal(infer_conso, label_conso))
            # post process
            total_skip += batch[2]["skip"]
            pbar.update(len(batch_image)+batch[2]["skip"])
        pbar.close()
        # result
        loss_graph /= train_iterator.num_samples - total_skip
        loss_vowel /= train_iterator.num_samples - total_skip
        loss_conso /= train_iterator.num_samples - total_skip
        loss /= train_iterator.num_samples - total_skip
        accuracy_graph /= train_iterator.num_samples - total_skip
        accuracy_vowel /= train_iterator.num_samples - total_skip
        accuracy_conso /= train_iterator.num_samples - total_skip
        score = (accuracy_graph * 2 + accuracy_vowel + accuracy_conso) / 4
        print(f"total_loss={loss:.4f}, total_score={score:.4f}")
        print(f"Accuracy")
        print(f"    grapheme={accuracy_graph:.4f}, vowel={accuracy_vowel:.4f}, consonant={accuracy_conso:.4f}")

        """ validation """
        print("validation")
        with torch.no_grad():
            total_skip = 0
            loss_graph = 0
            loss_vowel = 0
            loss_conso = 0
            loss = 0
            accuracy_graph = 0
            accuracy_vowel = 0
            accuracy_conso = 0
            pbar = tqdm.tqdm(total=validation_iterator.num_samples, leave=True, unit="samples")
            for batch in validation_iterator:
                # initialize
                optimizer.zero_grad()
                # input
                batch_image = torch.from_numpy(batch[0]).to(device)
                # infer
                outputs_graph, outputs_vowel, outputs_conso = net(batch_image)
                # label
                label_graph = torch.from_numpy(batch[1][:, 0]).to(device)
                label_vowel = torch.from_numpy(batch[1][:, 1]).to(device)
                label_conso = torch.from_numpy(batch[1][:, 2]).to(device)
                # calc loss
                batch_loss_graph = loss_function_graph(outputs_graph, label_graph)
                batch_loss_vowel = loss_function_vowel(outputs_vowel, label_vowel)
                batch_loss_conso = loss_function_conso(outputs_conso, label_conso)
                batch_loss = batch_loss_graph + batch_loss_vowel + batch_loss_conso
                # accumulate loss
                loss_graph += batch_loss_graph.item() * len(batch_image)
                loss_vowel += batch_loss_vowel.item() * len(batch_image)
                loss_conso += batch_loss_conso.item() * len(batch_image)
                loss += batch_loss.item() * len(batch_image)
                # infer result
                infer_graph = np.argmax(outputs_graph.cpu().detach().numpy(), axis=-1)
                infer_vowel = np.argmax(outputs_vowel.cpu().detach().numpy(), axis=-1)
                infer_conso = np.argmax(outputs_conso.cpu().detach().numpy(), axis=-1)
                # label
                label_graph = label_graph.cpu().detach().numpy()
                label_vowel = label_vowel.cpu().detach().numpy()
                label_conso = label_conso.cpu().detach().numpy()
                # score
                accuracy_graph += np.count_nonzero(np.equal(infer_graph, label_graph))
                accuracy_vowel += np.count_nonzero(np.equal(infer_vowel, label_vowel))
                accuracy_conso += np.count_nonzero(np.equal(infer_conso, label_conso))
                # post process
                total_skip += batch[2]["skip"]
                pbar.update(len(batch_image)+batch[2]["skip"])
            pbar.close()
            # result
            loss_graph /= validation_iterator.num_samples - total_skip
            loss_vowel /= validation_iterator.num_samples - total_skip
            loss_conso /= validation_iterator.num_samples - total_skip
            loss /= validation_iterator.num_samples - total_skip
            accuracy_graph /= validation_iterator.num_samples - total_skip
            accuracy_vowel /= validation_iterator.num_samples - total_skip
            accuracy_conso /= validation_iterator.num_samples - total_skip
            score = (accuracy_graph * 2 + accuracy_vowel + accuracy_conso) / 4
            print(f"total_loss={loss:.4f}, total_score={score:.4f}")
            print(f"Accuracy")
            print(f"    grapheme={accuracy_graph:.4f}, vowel={accuracy_vowel:.4f}, consonant={accuracy_conso:.4f}")

        if epoch % 10 == 9:
            torch.save(net.state_dict(), f"./result/epoch_{epoch:04d}.model")



if __name__ == "__main__":
    main()