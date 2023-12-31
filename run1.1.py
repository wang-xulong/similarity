"""
2 phase continual learning
"""
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models.resnet import resnet18
from argparse import Namespace
from metrics import compute_acc_fgt
from util import get_Cifar10, train_es, test
from external_libs.pyhessian import hessian
# ------------------------------------ step 0/5 : initialise hyper-parameters ------------------------------------
config = Namespace(
    project_name='CIFAR10',
    basic_task=0,  # count from 0
    experience=5,
    train_bs=16,
    test_bs=128,
    lr_init=0.1,
    max_epoch=5,
    run_times=2,
    patience=10,
    hessian=True,
    device='mps'
)

accuracy_list1 = []  # multiple run
accuracy_list2 = []
accuracy_list3 = []
accuracy_list4 = []

# use GPU?
no_cuda = False
use_cuda = not no_cuda and torch.backends.mps.is_available()
config.device = torch.device(config.device if torch.backends.mps.is_available() else "cpu")
kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

for run in range(config.run_times):
    print("run time: {}".format(run + 1))
    # ------------------------------------ step 1/5 : load data------------------------------------
    train_stream, test_stream = get_Cifar10()
    task_id = [1, 2, 3, 4, 5]
    # ------------------------------------ step 2/5 : define network-------------------------------
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    # ------------------------------------ step 3/5 : define loss function and optimization ------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch)
    # ------------------------------------ step 4/5 : training --------------------------------------------------
    # training basic task
    print("training basic task %d" % task_id[config.basic_task])
    basic_task_data = train_stream[config.basic_task]
    basic_task_test_data = test_stream[config.basic_task]
    model, _, _, avg_valid_losses, _, _ = train_es(basic_task_data, basic_task_test_data, model, criterion, optimizer,
                                                   scheduler, config.max_epoch, config.device, patience=config.patience,
                                                   task_id=config.basic_task, func_sim=False)

    # setting stage 1 matrix
    acc_array1 = np.zeros((4, 2))
    # testing basic task
    _, acc_array1[:, 0] = test(test_stream[config.basic_task], model, criterion, config.device, task_id=config.basic_task)
    # pop the src data from train_stream and test_stream
    train_stream.pop(config.basic_task)
    test_stream.pop(config.basic_task)
    task_id.pop(config.basic_task)
    # test other tasks except basic task
    for i, probe_data in enumerate(test_stream):
        with torch.no_grad():
            _, acc_array1[i, 1] = test(probe_data, model, criterion, config.device, task_id=task_id[i])
    # save task 1
    PATH = "./"
    trained_model_path = os.path.join(PATH, 'basic_model.pth')
    torch.save(model.state_dict(), trained_model_path)

    # setting stage 2 matrix
    acc_array2 = np.zeros((4, 2))
    for j, (train_data, test_data) in enumerate(zip(train_stream, test_stream)):
        print("task {} starting...".format(task_id[j]))
        # load old task's model
        trained_model = resnet18()
        trained_model.fc = nn.Linear(trained_model.fc.in_features, 2)  # final output dim = 2
        trained_model.load_state_dict(torch.load(trained_model_path))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(trained_model.parameters(), lr=config.lr_init, momentum=0.9, dampening=0.1)
        # training other tasks
        trained_model, _, _, _, _, new_task_first_loss = train_es(train_data, test_data, trained_model, criterion,
                                                                  optimizer, scheduler, config.max_epoch, config.device,
                                                                  config.patience, task_id=task_id[j], func_sim=True)

        # test model on basic task and task j
        with torch.no_grad():
            task_id = task_id[j]
            _, acc_array2[j, 0] = test(basic_task_test_data, trained_model, criterion, config.device, task_id=task_id)
            _, acc_array2[j, 1] = test(test_stream[j], trained_model, criterion, config.device, task_id=task_id)
        # (optional) calculate Hessian
        hessian_comp = hessian(trained_model, criterion, dataloader=test_stream[j], cuda=True)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
        print("The top Hessian eigenvalue of this model is %.4f" % top_eigenvalues[-1])

        # computing avg_acc and CF
    accuracy_list1.append([acc_array1[0, :], acc_array2[0, :]])
    accuracy_list2.append([acc_array1[1, :], acc_array2[1, :]])
    accuracy_list3.append([acc_array1[2, :], acc_array2[2, :]])
    accuracy_list4.append([acc_array1[3, :], acc_array2[3, :]])

accuracy_array1 = np.array(accuracy_list1)
accuracy_array2 = np.array(accuracy_list2)
accuracy_array3 = np.array(accuracy_list3)
accuracy_array4 = np.array(accuracy_list4)


avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array1)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array2)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array3)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array4)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
