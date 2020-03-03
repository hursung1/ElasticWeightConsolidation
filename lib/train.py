import torch
import torchvision
import numpy as np

import lib

def FineTuning(**kwargs):
    """
    Continual Learning with just Fine Tuning
    """
    dataloader = kwargs['dataloader']
    epochs = kwargs['epochs']
    optim = kwargs['optim']
    crit = kwargs['crit']
    net = kwargs['net']

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optim.zero_grad()
            outputs = net(x)
            loss = crit(outputs, y)
            loss.backward()
            optim.step()
            running_loss += loss.item()

        if epoch % 10 == 9:
            print("[Epoch %d/%d] Loss: %.3f"%(epoch+1, epochs, running_loss))


def L2Learning(**kwargs):
    """
    Continual Learning with L2 Regularization Term
    """
    past_task_params = kwargs['past_task_params']
    dataloader = kwargs['dataloader']
    epochs = kwargs['epochs']
    optim = kwargs['optim']
    crit = kwargs['crit']
    net = kwargs['net']
    ld = kwargs['ld']

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optim.zero_grad()
            outputs = net(x)
            loss = crit(outputs, y)

            reg = 0.0
            for past_param in past_task_params:
                for i, param in enumerate(net.parameters()):
                    penalty = (past_param[i] - param) ** 2
                    reg += penalty.sum()
                loss += reg * (ld / 2)

            loss.backward()
            optim.step()
            running_loss += loss.item()

        if epoch % 10 == 9:
            print("[Epoch %d/%d] Loss: %.3f"%(epoch+1, epochs, running_loss))

    ### Save parameters to use next task learning
    tensor_param = []
    for params in net.parameters():
        tensor_param.append(params.detach().clone())
    '''
    tensor_param = torch.stack(tensor_param)

    if past_task_params.nelement() > 0:
        past_task_params = torch.cat((past_task_params, tensor_param.unsqueeze(0)))
    else:
        past_task_params = tensor_param.unsqueeze(0)
    '''
    past_task_params.append(tensor_param)


def EWCLearning():
    """
    Continual Learning with Fisher Regularization Term
    """
    past_task_params = kwargs['past_task_params']
    past_fisher_mat = kwargs['past_fisher_mat']
    dataloader = kwargs['dataloader']
    epochs = kwargs['epochs']
    optim = kwargs['optim']
    crit = kwargs['crit']
    net = kwargs['net']
    ld = kwargs['ld']

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            optim.zero_grad()
            outputs = net(x)
            loss = crit(outputs, y)

            reg = 0.0
            for task, past_param in enumerate(past_task_params):
                for i, param in net.parameters():
                    penalty = (past_param[i] - param) ** 2
                    penalty *= past_fisher_mat[task]
                    reg += penalty.sum()
                loss += reg * (ld / 2)

            loss.backward()
            optim.step()
            running_loss += loss.item()

        if epoch % 10 == 9:
            print("[Epoch %d/%d] Loss: %.3f"%(epoch+1, epochs, running_loss))

    ### Save parameters to use at next task learning
    tensor_param = []
    for params in net.parameters():
        tensor_param.append(params.detach().clone())
    tensor_param = torch.stack(tensor_param)
    past_task_params = torch.cat((past_task_params, tensor_param.unsqueeze(0)))

    ### Save Fisher matrix
    FisherMatrix = lib.get_fisher(net, crit, dataloader)
    past_fisher_mat = torch.cat((past_fisher_mat, FisherMatrix.unsqueeze(0)))