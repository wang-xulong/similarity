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
from external_libs.hessian_eigenthings import compute_hessian_eigenthings
import wandb
import datetime

# ------------------------------------ step 0/5 : initialise hyper-parameters ------------------------------------
config = Namespace(
    project_name='CIFAR10',
    basic_task_index=0,  # count from 0
    experience=5,
    train_bs=16,
    test_bs=128,
    lr_init=0.01,
    max_epoch=200,
    run_times=5,
    patience=100,
    hessian=True,
    device='cuda',
    kwargs={}
)

accuracy_list1 = []  # multiple run
accuracy_list2 = []
accuracy_list3 = []
accuracy_list4 = []
# fill in hessian
hessian_result = {}

# use GPU?
no_cuda = False
# mps
# use_cuda = not no_cuda and torch.backends.mps.is_available()
# config.device = torch.device(config.device if torch.backends.mps.is_available() else "cpu")
# HPC
use_cuda = not no_cuda and torch.cuda.is_available()
config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
config.kwargs = {"num_workers": 16, "pin_memory": True, "prefetch_factor": config.train_bs * 2} if use_cuda else {}

now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
wandb.init(project=config.project_name, config=config.__dict__, name=now_time, save_code=True)

for run in range(config.run_times):
    print("run time: {}".format(run + 1))
    # ------------------------------------ step 1/5 : load data------------------------------------
    train_stream, test_stream = get_Cifar10(config)
    task_id = [1, 2, 3, 4, 5]
    # ------------------------------------ step 2/5 : define network-------------------------------
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    # ------------------------------------ step 3/5 : define loss function and optimization ------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch)
    # ------------------------------------ step 4/5 : training --------------------------------------------------
    # training basic task
    print("training basic task %d" % task_id[config.basic_task_index])
    basic_task_data = train_stream[config.basic_task_index]
    basic_task_test_data = test_stream[config.basic_task_index]
    model, _, _, avg_valid_losses, _, _ = train_es(basic_task_data, basic_task_test_data, model, criterion, optimizer,
                                                   scheduler, config.max_epoch, config.device, patience=config.patience,
                                                   task_id=config.basic_task_index, func_sim=False)
    # (option) calculate Hessian eignvalues
    eigenvals, _ = compute_hessian_eigenthings(
        model,
        basic_task_test_data,
        criterion,
        num_eigenthings=3,
        mode="power_iter",
        max_possible_gpu_samples=2048,
        use_gpu=True,
    )
    key = "run {} - task {}‘s Hessian".format(run+1, task_id[config.basic_task_index])
    hessian_result[key] = eigenvals
    print(key, hessian_result[key])

    # setting stage 1 matrix
    acc_array1 = np.zeros((4, 2))
    # testing basic task
    _, acc_array1[:, 0] = test(test_stream[config.basic_task_index], model, criterion, config.device, task_id=config.basic_task_index)
    # pop the src data from train_stream and test_stream
    train_stream.pop(config.basic_task_index)
    test_stream.pop(config.basic_task_index)
    task_id.pop(config.basic_task_index)
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
            _, acc_array2[j, 0] = test(basic_task_test_data, trained_model, criterion, config.device, task_id=task_id[j])
            _, acc_array2[j, 1] = test(test_stream[j], trained_model, criterion, config.device, task_id=task_id[j])

        # (option) calculate Hessian eignvalues
        eigenvals, _ = compute_hessian_eigenthings(
            trained_model,
            test_stream[j],
            criterion,
            num_eigenthings=3,
            mode="power_iter",
            max_possible_gpu_samples=2048,
            use_gpu=True,
        )
        key = "run {} - task {}‘s Hessian".format(run + 1, task_id[j])
        hessian_result[key] = eigenvals
        print(key, hessian_result[key])

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

# (optional) save hessian result
np.save('hessian_result.npy', hessian_result)
wandb.finish()