from __future__ import print_function, division

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import datasets, models, transforms

from torch.autograd import Variable

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy

import onnx
import tensorrt
import onnx_tensorrt.backend as backend

model = onnx.load("resnet18.onnx")
engine = backend.prepare(model, device='CUDA:0', max_batch_size=64)
input_data = np.random.random(size=(1, 3, 224, 224)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-180,180)),
        transforms.RandomAffine(degrees=(-30,30), shear=(-20,20)),
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5548, 0.4508, 0.3435], [0.2281, 0.2384, 0.2376])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5604, 0.4540, 0.3481], [0.2260, 0.2367, 0.2352])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5604, 0.4540, 0.3481], [0.2260, 0.2367, 0.2352])
    ]),
}

data_dir = '../data/food/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
test_batch = 64
x = 'test'
dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=test_batch,
                                             shuffle=True, num_workers=0)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#device = torch.device("cpu")

classes = ('Bread', 'DairyProduct', 'Dessert', 'Egg', 'Friedfood',
    'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit')

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 11)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.5 every 10 epochs
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, mode='min', patience=6, verbose=True)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)

def test_tensorrt(model, t_data):
    print('test data set: %s' %(t_data))
    
    device = torch.device("cuda:0")
    correct = 0
    total = 0
    running_loss = 0.0

    class_correct = list(0. for i in range(11))
    class_total = list(0. for i in range(11))

    with torch.no_grad():
        for data in dataloaders[t_data]:
            images, labels = data
 
            # images = images.to(device)
            # labels = labels.to(device)
            
            images = images.numpy().astype(np.float32)
            outputs = model.run(images)[0]
            outputs = torch.from_numpy(outputs)
           
            _, predicted = torch.max(outputs, 1)
        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            c = (predicted == labels).squeeze()

            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the %d test images: %.2f%%, and loss is: %.3f'
          % (total, 100 * correct / total, running_loss / total))

    for i in range(11):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

for test_batch in [2, 4, 8, 16, 32, 64, 128, 256]:
    x = 'test'
    dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=test_batch,
                                                 shuffle=True, num_workers=0)
    
    model = onnx.load("resnet18.onnx")
    engine = backend.prepare(model, device='CUDA:0', max_batch_size=test_batch)

    tensorrt_model = engine
    print('batch size: {}'.format(test_batch))
    t0 = time.time()
    test_tensorrt(tensorrt_model, 'test')
    print('{} seconds'.format(time.time() - t0))

