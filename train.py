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
    net = models.bengali.BengaliVGG16([168, 11, 7]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(100):

        print(f"[epoch {epoch:04d}]")

        """ train """
        total_skip = 0
        loss_a = 0
        loss_b = 0
        loss_c = 0
        loss = 0
        correct_a = 0
        correct_b = 0
        correct_c = 0
        correct_all = 0
        pbar = tqdm.tqdm(total=train_iterator.num_samples)
        pbar.set_description("training")
        for batch in train_iterator:
            # initialize
            optimizer.zero_grad()
            # input
            batch_image = torch.Tensor(batch[0]).to(device)
            # infer
            outputs_a, outputs_b, outputs_c = net(batch_image)
            # to cpu
            outputs_a = outputs_a.cpu()
            outputs_b = outputs_b.cpu()
            outputs_c = outputs_c.cpu()
            label = np.array(batch[1]).astype(np.int)
            label_a = torch.from_numpy(label[:, 0]).cpu()
            label_b = torch.from_numpy(label[:, 1]).cpu()
            label_c = torch.from_numpy(label[:, 2]).cpu()
            # calc loss
            batch_loss_a = torch.nn.functional.cross_entropy(outputs_a, label_a)
            batch_loss_b = torch.nn.functional.cross_entropy(outputs_b, label_b)
            batch_loss_c = torch.nn.functional.cross_entropy(outputs_c, label_c)
            batch_loss = batch_loss_a + batch_loss_b + batch_loss_c
            # backward
            batch_loss.backward()
            # optimize
            optimizer.step()
            # accumulate loss
            loss_a += batch_loss_a / len(batch_image)
            loss_b += batch_loss_b / len(batch_image)
            loss_c += batch_loss_c / len(batch_image)
            loss += batch_loss / len(batch_image)
            # accumulate correct
            outputs_a = outputs_a.detach().numpy()
            outputs_b = outputs_b.detach().numpy()
            outputs_c = outputs_c.detach().numpy()
            infer_a = np.argmax(outputs_a, axis=-1)
            infer_b = np.argmax(outputs_b, axis=-1)
            infer_c = np.argmax(outputs_c, axis=-1)
            label_a = label_a.detach().numpy()
            label_b = label_b.detach().numpy()
            label_c = label_c.detach().numpy()
            batch_correct_a = np.equal(infer_a, label_a)
            batch_correct_b = np.equal(infer_b, label_b)
            batch_correct_c = np.equal(infer_c, label_c)
            batch_correct_all = np.logical_and(np.logical_and(batch_correct_a,
                                                              batch_correct_b),
                                               batch_correct_c)
            correct_a += np.count_nonzero(batch_correct_a)
            correct_b += np.count_nonzero(batch_correct_b)
            correct_c += np.count_nonzero(batch_correct_c)
            correct_all += np.count_nonzero(batch_correct_all)
            # post process
            total_skip += batch[2]["skip"]
            pbar.update(len(batch_image)+batch[2]["skip"])
        pbar.close()
        correct_a /= train_iterator.num_samples - total_skip
        correct_b /= train_iterator.num_samples - total_skip
        correct_c /= train_iterator.num_samples - total_skip
        correct_all /= train_iterator.num_samples - total_skip
        print(f"result : loss={loss:.4f}, root_loss={loss_a:.4f}, vowel_loss={loss_b:.4f}, consonantloss={loss_c:.4f}")
        print(f"         acc={correct_all:.4f}, root_acc={correct_a:.4f}, vowel_acc={correct_b:.4f}, consonant_acc={correct_c:.4f}")
        

        with torch.no_grad(): 
            """ validation """
            total_skip = 0
            loss_a = 0
            loss_b = 0
            loss_c = 0
            loss = 0
            correct_a = 0
            correct_b = 0
            correct_c = 0
            correct_all = 0
            pbar = tqdm.tqdm(total=validation_iterator.num_samples)
            pbar.set_description("validation")
            for batch in validation_iterator:
                # initialize
                optimizer.zero_grad()
                # input
                batch_image = torch.Tensor(batch[0]).to(device)
                # infer
                outputs_a, outputs_b, outputs_c = net(batch_image)
                # to cpu
                outputs_a = outputs_a.cpu()
                outputs_b = outputs_b.cpu()
                outputs_c = outputs_c.cpu()
                label = np.array(batch[1]).astype(np.int)
                label_a = torch.from_numpy(label[:, 0]).cpu()
                label_b = torch.from_numpy(label[:, 1]).cpu()
                label_c = torch.from_numpy(label[:, 2]).cpu()
                # calc loss
                batch_loss_a = torch.nn.functional.cross_entropy(outputs_a, label_a)
                batch_loss_b = torch.nn.functional.cross_entropy(outputs_b, label_b)
                batch_loss_c = torch.nn.functional.cross_entropy(outputs_c, label_c)
                batch_loss = batch_loss_a + batch_loss_b + batch_loss_c
                # accumulate loss
                loss_a += batch_loss_a / len(batch_image)
                loss_b += batch_loss_b / len(batch_image)
                loss_c += batch_loss_c / len(batch_image)
                loss += batch_loss / len(batch_image)
                # accumulate correct
                outputs_a = outputs_a.detach().numpy()
                outputs_b = outputs_b.detach().numpy()
                outputs_c = outputs_c.detach().numpy()
                infer_a = np.argmax(outputs_a, axis=-1)
                infer_b = np.argmax(outputs_b, axis=-1)
                infer_c = np.argmax(outputs_c, axis=-1)
                label_a = label_a.detach().numpy()
                label_b = label_b.detach().numpy()
                label_c = label_c.detach().numpy()
                batch_correct_a = np.equal(infer_a, label_a)
                batch_correct_b = np.equal(infer_b, label_b)
                batch_correct_c = np.equal(infer_c, label_c)
                batch_correct_all = np.logical_and(np.logical_and(batch_correct_a,
                                                                  batch_correct_b),
                                                   batch_correct_c)
                correct_a += np.count_nonzero(batch_correct_a)
                correct_b += np.count_nonzero(batch_correct_b)
                correct_c += np.count_nonzero(batch_correct_c)
                correct_all += np.count_nonzero(batch_correct_all)
               # post process
                total_skip += batch[2]["skip"]
                pbar.update(len(batch_image)+batch[2]["skip"])
            pbar.close()
            correct_a /= validation_iterator.num_samples - total_skip
            correct_b /= validation_iterator.num_samples - total_skip
            correct_c /= validation_iterator.num_samples - total_skip
            correct_all /= validation_iterator.num_samples - total_skip
            print(f"result : loss={loss:.4f}, root_loss={loss_a:.4f}, vowel_loss={loss_b:.4f}, consonantloss={loss_c:.4f}")
            print(f"         acc={correct_all:.4f}, root_acc={correct_a:.4f}, vowel_acc={correct_b:.4f}, consonant_acc={correct_c:.4f}")

        torch.save(net.state_dict(), f"./result/epoch_{epoch:04d}.model")



if __name__ == "__main__":
    main()