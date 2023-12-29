from metrics import accuracy
from pytorchtools import EarlyStopping
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import copy
from CLDataset import MyDataset
import wandb

experience = 5


def trainES(train_data, test_data, model, criterion, optimizer, max_epoch, device, patience, task_id,
            func_sim=False):
    # 要记录训练的epoch
    record_epoch = 0
    # 新任务第一次梯度下降后的损失
    new_task_loss = -1
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

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for e in range(max_epoch):
        model.train()
        for k, batch in enumerate(train_data):  # 对当前批次数据取出batch数据并开始训练model
            # 获取数据与标签
            x_train, y_train = batch[0].to(device), batch[1].to(device)
            model.to(device)
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if k == 0 and func_sim:
            #     new_task_first_loss = loss.item()
            # RECORD training loss
            train_losses.append(loss.item())
            train_accs.append(accuracy(y_pred, y_train).item())

        if func_sim is True and e == record_epoch:
            new_task_loss = copy.deepcopy(train_losses[1])  # 第1次为模型梯度下降并更新一次后的损失

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
                # 不再记录每batch的训练情况
                # wandb.log({"Train id:" + str(task_id): accuracy(y_pred, y_test).item()})

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            train_acc = np.average(train_accs)
            valid_acc = np.average(valid_accs)
            # 记录每个epoch的训练下任务的情况
            wandb.log({"Train id:" + str(task_id): valid_acc, "epoch": e})
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            avg_train_accs.append(train_acc)
            avg_valid_accs.append(valid_acc)
            epoch_len = len(str(max_epoch))
            print_msg = (f'[{e:>{epoch_len}}/{max_epoch:>{epoch_len}}] ' +
                         f' train_loss: {train_loss:.5f}' +
                         f' valid_loss: {valid_loss:.5f}' +
                         f' train_acc: {train_acc:.5f}' +
                         f' valid_acc: {valid_acc:.5f}')
            print(print_msg)
            train_losses = []
            valid_losses = []
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_train_accs, avg_valid_losses, avg_valid_accs, new_task_loss


def train(train_data, test_data, model, criterion, optimizer, max_epoch, device, func_sim=False):
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
        for k, batch in enumerate(train_data):  # 对当前批次数据取出batch数据并开始训练model
            # 获取数据与标签
            x_train, y_train = batch[0].to(device), batch[1].to(device)
            model.to(device)
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if k == 1 and func_sim:
                new_task_first_loss = loss.item()
            # RECORD training loss
            train_losses.append(loss.item())
            train_accs.append(accuracy(y_pred, y_train).item())
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

    return model, avg_train_losses, avg_train_accs, avg_valid_losses, avg_valid_accs, new_task_first_loss


def test(test_data, model, criterion, device, task_id):
    model.eval()
    test_data_loss = 0
    test_data_acc = 0
    for t_j, test_batch in enumerate(test_data):
        x_test, y_test = test_batch[0].to(device), test_batch[1].to(device)
        model.to(device)
        test_pred = model(x_test)
        loss = criterion(test_pred, y_test)
        test_data_loss += loss * test_data.batch_size
        current_acc = accuracy(test_pred, y_test)
        test_data_acc += current_acc

    test_loss = test_data_loss / len(test_data.dataset)
    acc = test_data_acc / (t_j + 1)
    print("-----------test loss {:.4}, acc {:.4} ".format(test_loss, acc))
    wandb.log({"Test id:" + str(task_id): acc})
    return test_loss.cpu(), acc.cpu()


def get_Cifar10(train_bs=128, test_bs=64):
    train_dir = os.path.join("./", "Data", "SplitCifar10", "train")
    test_dir = os.path.join("./", "Data", "SplitCifar10", "test")
    train_stream = []
    test_stream = []
    # MNIST 数据集处理
    trainTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    testTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    # 构建CLMyDataset实例
    for e in range(experience):
        # 确定当前context的数据源路径
        train_txt_path = train_dir + ' ' + str(e) + '.txt'
        test_txt_path = test_dir + ' ' + str(e) + '.txt'
        # 创建Dataset实例
        train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
        test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)
        # 构建CLDataLoader
        train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_workers=2, pin_memory=True,
                                  prefetch_factor=train_bs * 2)
        test_loader = DataLoader(dataset=test_data, batch_size=test_bs, num_workers=2, pin_memory=True,
                                 prefetch_factor=test_bs * 2)
        # 添加到stream list中
        train_stream.append(train_loader)
        test_stream.append(test_loader)
    return train_stream, test_stream


def get_Cifar100(train_bs=128, test_bs=64):
    base_dir = "/home/acq21xw"
    train_dir = os.path.join(base_dir, "Data", "SplitCifar100_2class_v3", "train")
    test_dir = os.path.join(base_dir, "Data", "SplitCifar100_2class_v3", "test")
    train_stream = []
    test_stream = []
    # MNIST 数据集处理
    trainTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    testTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    # 构建CLMyDataset实例
    for e in range(experience):
        # 确定当前context的数据源路径
        train_txt_path = train_dir + ' ' + str(e) + '.txt'
        test_txt_path = test_dir + ' ' + str(e) + '.txt'
        # 创建Dataset实例
        train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
        test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)
        # 构建CLDataLoader
        train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_workers=2, pin_memory=True,
                                  prefetch_factor=train_bs * 2)
        test_loader = DataLoader(dataset=test_data, batch_size=test_bs, num_workers=2, pin_memory=True,
                                 prefetch_factor=test_bs * 2)
        # 添加到stream list中
        train_stream.append(train_loader)
        test_stream.append(test_loader)
    return train_stream, test_stream


def get_MNIST(train_bs=128, test_bs=128):
    train_dir = os.path.join("Data", "SplitMNIST", "train")
    test_dir = os.path.join("Data", "SplitMNIST", "test")
    train_stream = []
    test_stream = []
    # MNIST 数据集处理
    trainTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    testTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    # 构建CLMyDataset实例
    for e in range(experience):
        # 确定当前context的数据源路径
        train_txt_path = train_dir + ' ' + str(e) + '.txt'
        test_txt_path = test_dir + ' ' + str(e) + '.txt'
        # 创建Dataset实例
        train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
        test_data = MyDataset(txt_path=test_txt_path, transform=testTransform)
        # 构建CLDataLoader
        train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=test_bs)
        # 添加到stream list中
        train_stream.append(train_loader)
        test_stream.append(test_loader)
    return train_stream, test_stream
