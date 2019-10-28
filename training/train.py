from model.FCN import FCN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from dataloader import SugarBeetDataset
import time


n_class    = 3
batch_size = 1
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
num_workers = 4 # if on vscode set to 0
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)



use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

_transform = transforms.ToTensor()

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'

    # our dataset has three classes only - background, weed and crop
    num_classes = 3

    # use our dataset and defined transformations
    dataset = SugarBeetDataset.from_config()

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()

    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
    dataset = torch.utils.data.Subset(dataset, indices[:-50])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=num_workers)

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

        # train for one epoch, printing every 10 iterations
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            print("loss:", loss.item())

            loss.backward()
            optimizer.step()

        # update the learning rate
        scheduler.step()



if __name__ == "__main__":
    main()
