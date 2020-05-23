import torch
import torchvision
import numpy as np

import pyfiles.PMNISTDataLoader as DataLoader

def setPMNISTDataLoader(num_task, batch_size):
    """

    Returns
    -------------
    List of Permuted MNIST train/test Dataloaders
    """
    train_loader = []
    test_loader = []

    for i in range(num_task):
        shuffle_seed = np.arange(28*28)
        np.random.shuffle(shuffle_seed)
        train_loader.append(torch.utils.data.DataLoader(
            DataLoader.PermutedMNISTDataLoader(
                train=True, 
                shuffle_seed=shuffle_seed),
            batch_size=batch_size))
        
        test_loader.append(torch.utils.data.DataLoader(
            DataLoader.PermutedMNISTDataLoader(
                train=False, 
                shuffle_seed=shuffle_seed),
            batch_size=batch_size))
    
    return train_loader, test_loader


def get_fisher(net, crit, dataloader):
    FisherMatrix = []
    net.eval()
    for params in net.parameters():
        if params.requires_grad:
            ZeroMat = torch.zeros_like(params)
            FisherMatrix.append(ZeroMat)

    #FisherMatrix = torch.stack(FisherMatrix)

    for data in dataloader:
        x, y = data
        x = x.view(-1, 1, 28, 28)
        num_data = x.shape[0]
        if torch.cuda.is_available():
            x = x.cuda(3)
            y = y.cuda(3)

        net.zero_grad()
        outputs = net(x)
        loss = crit(outputs, y)
        loss.backward()

        for i, params in enumerate(net.parameters()):
            if params.requires_grad:
                FisherMatrix[i] += params.grad.data ** 2 / num_data

    return FisherMatrix