import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import pyfiles.PMNISTDataLoader as DataLoader
import pyfiles.models as models
import pyfiles.train as train
import pyfiles.eval as evalu
import pyfiles.lib as lib

num_task = 10
batch_size = 64
hidden_layer_num = 400
epochs = 20
train_loader, test_loader = lib.setPMNISTDataLoader(num_task, batch_size)

### Fine-Tuning
'''

net = models.FCNetwork(hidden_layer_num)
crit = torch.nn.CrossEntropyLoss()
if torch.cuda.is_available():
    net = net.cuda()
    crit = crit.cuda()

optim = torch.optim.SGD(net.parameters(), lr=0.001)    
plain_acc = {}
for t in range(num_task):
    train.FineTuning(
        dataloader = train_loader[t],
        epochs = epochs, 
        optim = optim,
        crit = crit,
        net = net,
    )
    
    each_task_acc, acc_mean = evalu.eval(
        dataloader = test_loader,
        num_task = t,
        net = net
    )
    print(each_task_acc)
    plain_acc[t] = acc_mean

### L2 Regularization
ld_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
l2_acc = {}

for ld in ld_list:
    print("Lambda %f"%(ld))
    net = models.FCNetwork(hidden_layer_num)
    crit = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
        crit = crit.cuda()

    past_task_params = []
    optim = torch.optim.SGD(net.parameters(), lr=0.001)    
    l2_acc_per_ld = {}
    for t in range(num_task):
        train.L2Learning(
            past_task_params = past_task_params,
            dataloader = train_loader[t],
            epochs = epochs,
            optim = optim,
            crit = crit,
            net = net,
            ld = ld
        )

        each_task_acc, acc_mean = evalu.eval(
            dataloader = test_loader,
            num_task = t,
            net = net
        )
        print(each_task_acc)
        l2_acc_per_ld[t] = acc_mean
        
    l2_acc[ld] = l2_acc_per_ld
'''
### Elastic Weight Consolidation
ld_list=[10, 30, 100, 300, 1000, 3000]
ewc_acc = {}

for ld in ld_list:
    print("Lambda %f"%(ld))
    net = models.FCNetwork(hidden_layer_num)
    crit = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        net = net.cuda()
        crit = crit.cuda()

    past_task_params = []
#     past_fisher_mat = torch.zeros(0)
    past_fisher_mat=[]
    optim = torch.optim.Adam(net.parameters(), lr=0.001)    
    ewc_acc_per_ld = {}
    for t in range(num_task):
        train.EWCLearning(
            past_task_params = past_task_params,
            past_fisher_mat = past_fisher_mat,
            dataloader = train_loader[t],
            epochs = epochs,
            optim = optim,
            crit = crit,
            net = net,
            ld = ld
        )

        print(past_task_params)
        each_task_acc, acc_mean = evalu.eval(
            dataloader = test_loader,
            num_task = t,
            net = net
        )
        print(each_task_acc)
        ewc_acc_per_ld[t] = acc_mean
        
    ewc_acc[ld] = ewc_acc_per_ld