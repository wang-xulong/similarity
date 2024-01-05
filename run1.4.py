"""
calculate new task gradient, base on cifar10
"""
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision.models.resnet import resnet18
from argparse import Namespace
from metrics import compute_acc_fgt
from util import get_Cifar10, train_es, test, save_checkpoint
from metrics import accuracy

watch_epoch = 0


def _train(train_data, test_data, model, criterion, optimizer, max_epoch, device, task_id):
    gradient_norm = []
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    train_accs = []
    valid_accs = []
    avg_train_accs = []
    avg_valid_accs = []
    new_task_first_loss = float('inf')

    for e in range(max_epoch):
        model.train()
        norm_squared = torch.tensor(0.0).to(device)
        for k, batch in enumerate(train_data):  # 对当前批次数据取出batch数据并开始训练model
            # 获取数据与标签
            x_train, y_train = batch[0].to(device), batch[1].to(device)
            model.to(device)
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            # !获取梯度信息
            if e == watch_epoch:
                for _, v in model.named_parameters():
                    norm_squared += torch.sum(torch.square(v.grad))  # 计算l2^2
            optimizer.step()
            # RECORD training loss
            train_losses.append(loss.item())
            train_accs.append(accuracy(y_pred, y_train).item())
            if e == watch_epoch:
                gradient_result = torch.sqrt(norm_squared).item()
                gradient_norm.append(gradient_result)
        # validation
        model.eval()
        with torch.no_grad():
            for test_batch in test_data:
                x_test, y_test = test_batch[0].to(device), test_batch[1].to(device)
                model.to(device)
                y_pred = model(x_test)
                test_loss = criterion(y_pred, y_test)
                # record valid loss
                valid_losses.append(test_loss.item())
                valid_accs.append(accuracy(y_pred, y_test).item())
            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            train_acc = np.average(train_accs)
            valid_acc = np.average(valid_accs)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            avg_train_accs.append(train_acc)
            avg_valid_accs.append((valid_acc))
            epoch_len = len(str(max_epoch))
            print_msg = (f'[{e:>{epoch_len}}/{max_epoch:>{epoch_len}}] ' +
                         f' train_loss: {train_loss:.5f} ' +
                         f' valid_loss: {valid_loss:.5f}' +
                         f' train_acc: {train_acc:.5f}' +
                         f' valid_acc: {valid_acc:.5f}')
            print(print_msg)
            train_losses = []
            valid_losses = []
        if watch_epoch <= e:
            break

    return gradient_norm


# ------------------------------------ step 0/5 : initialise hyper-parameters ------------------------------------
config = Namespace(
    project_name='CIFAR10',
    basic_task_index=0,  # count from 0
    experience=5,
    train_bs=10,
    test_bs=128,
    lr_init=0.25,
    max_epoch=30,
    run_times=1,
    patience=20,
    hessian=False,
    device='cuda',
    kwargs={},
    func_sim=False,
    weights_norm=True
)

norms = {}

# use GPU?
no_cuda = False
# mps
# use_cuda = not no_cuda and torch.backends.mps.is_available()
# config.device = torch.device(config.device if torch.backends.mps.is_available() else "cpu")
# HPC
use_cuda = not no_cuda and torch.cuda.is_available()
config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
config.kwargs = {"num_workers": 16, "pin_memory": True, "prefetch_factor": config.train_bs * 2} if use_cuda else {}

# iterate new tasks
for task_index in range(1, 5):
    # load new task data
    train_stream, test_stream = get_Cifar10(config)
    train_data, test_data = train_stream[task_index], test_stream[task_index]
    # load basic model
    TRIAL_ID = "62BEF1"
    PATH = "./outputs/" + TRIAL_ID
    filename = '{PATH}/model-{TRIAL_ID}-{task}.pth'.format(PATH=PATH, TRIAL_ID=TRIAL_ID, task=1)
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)  # final output dim = 2
    model.load_state_dict(torch.load(filename))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr_init * 0.1, momentum=0.9, dampening=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.patience, gamma=0.1)
    key = 'task {}'.format(task_index+1)
    norms[key] = _train(train_data, test_data, model, criterion, optimizer,
                        config.max_epoch, config.device, task_index+1)

print(sum(norms['task 2'])/len(norms['task 2']))
print(sum(norms['task 3'])/len(norms['task 3']))
print(sum(norms['task 4'])/len(norms['task 4']))
print(sum(norms['task 5'])/len(norms['task 5']))
# 第一个batch 足以