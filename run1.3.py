"""
example: calculate the distance of tasks from weights norm
"""
from torchvision.models.resnet import resnet18
import torch.nn as nn
import torch


def calculate_weights_norm(model1, model2):
    # save norm result
    norm_squared = torch.tensor(0.0)
    # Iterate through all parameters in the model
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        norm_squared += torch.sum(param1 - param2) ** 2
    return torch.sqrt(norm_squared).item()


TRIAL_ID = "62BEF1"
PATH = "./outputs/"+TRIAL_ID

# load basic model
filename1 = '{PATH}/model-{TRIAL_ID}-{task}.pth'.format(PATH=PATH, TRIAL_ID=TRIAL_ID, task=1)
model1 = resnet18()
model1.fc = nn.Linear(model1.fc.in_features, 2)  # final output dim = 2
model1.load_state_dict(torch.load(filename1))

# load new models
filename2 = '{PATH}/model-{TRIAL_ID}-{task}.pth'.format(PATH=PATH, TRIAL_ID=TRIAL_ID, task=5)
model2 = resnet18()
model2.fc = nn.Linear(model2.fc.in_features, 2)  # final output dim = 2
model2.load_state_dict(torch.load(filename2))

# calculate the distance of tasks from weights norm
weights_norm = calculate_weights_norm(model1, model2)
# display
print(weights_norm)
