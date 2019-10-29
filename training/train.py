from model.FCN import FCN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from dataloader import SugarBeetDataset
from utils import intersectionAndUnion, accuracy, visualize
import time
import cv2
import numpy as np


n_class    = 3
batch_size = 1
epochs     = 1
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
num_workers = 0 # if on vscode set to 0
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)



use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

_transform = transforms.ToTensor()

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cpu'

    # our dataset has three classes only - background, weed and crop
    num_classes = 3

    # use our dataset and defined transformations
    dataset = SugarBeetDataset.from_config()

    """ MODEL TESTING SNIPPET"""
    test_input, test_target = dataset[0]
    test_input = test_input[None, ...].to(device)
    test_target = test_target[None, ...].to(device)

    model = FCN(n_class=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)

    for i in range(50000):
        optimizer.zero_grad()
        model.train()

        test_output = model(test_input)
        loss = criterion(test_output, test_target)

        print('Loss {}'.format(loss.item()))

        loss.backward()
        optimizer.step()

        if i%50==0:

            class_confidences = nn.functional.softmax(test_output, dim=1)

            cv_input = test_input.cpu().detach().numpy()[0].transpose(1, 2, 0)
            cv_sugar_beet_confidence = class_confidences.cpu().detach().numpy()[0, 2, ...]

            min_confidence = np.min(cv_sugar_beet_confidence)
            max_confidence = np.max(cv_sugar_beet_confidence)
            print('confidence min, max =', min_confidence, max_confidence)

            cv_sugar_beet_confidence = (255.0*(cv_sugar_beet_confidence-min_confidence/(max_confidence-min_confidence+0.0001))).astype(np.uint8)
            cv_target = test_target.cpu().detach().numpy()
            cv_target_sugar_beet = (255*(cv_target[0, ...]==2)).astype(np.uint8)
            # cv_target_sugar_beet = 50*(cv_target.transpose(1, 2, 0)).astype(np.uint8)

            print('target max =', np.max(cv_target))

            # print(cv_input.shape)
            # print(cv_sugar_beet_confidence.shape)

            cv2.imshow('taget sugar beet', cv_target_sugar_beet)
            cv2.imshow('input', cv_input)
            cv2.imshow('sugar beet', cv_sugar_beet_confidence)
            cv2.waitKey(1)

    exit()


    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()

    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
    dataset = torch.utils.data.Subset(dataset, indices[:-50])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=num_workers)


    # move model to the right device
    model = FCN(n_class=num_classes)
    model.to(device)

    # construct an optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)

    # and a learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        min_loss = 1.
        # train for one epoch, printing every 10 iterations
        for iter, batch in enumerate(data_loader):
            inputs, targets = batch

            optimizer.zero_grad()

            inputs = inputs.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            if loss.item() < min_loss:
                min_loss = loss.item()
                # torch.save(model.state_dict(), "ckpts/{}".format())

            if iter % 10 == 0:
                print("epochs:", epoch, "iterations:", iter, "loss:", loss.item())

            loss.backward()
            optimizer.step()

        # evaluation
        # TODO
        # for iter, batch in enumerate(data_loader_test):
        #     inputs, targets = batch
        #     inputs = inputs.to(device)
        #
        #     with torch.no_grad():
        #         outputs = model(inputs)


if __name__ == "__main__":
    main()
