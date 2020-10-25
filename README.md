# image-classifier train.py
Udacity Image classifier project
# Imports here

#Imports and Python torch Modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import json
import copy
from collections import OrderedDict

from PIL import Image
import glob, os

#Imports entries from Argparse
from get_input_args import get_input_args

# Function that checks command line arguments using in_arg  
in_arg = get_input_args()
print(in_arg)

#transform data for training the network
data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define your transforms for the training, validation, and testing sets

data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])


#Load the datasets with ImageFolder
data_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)


#Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(data_datasets, batch_size=40, shuffle=True)
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=40, shuffle=True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=40, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=40, shuffle=True)
    
#label mapping
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
        
#Building and training network
#Defining Hyperparameters
arch = in_arg.arch
epochs = in_arg.epochs
learning_rate = in_arg.learning_rate
hidden_units = in_arg.hidden_units
gpu = in_arg.gpu
hidden_output = [hidden_units, len(cat_to_name)]

if in_arg.gpu == True: 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else: device = torch.device("cpu")

#Model choosing
vgg13 = models.vgg13(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

models = {'vgg13': vgg13, 'vgg16': vgg16}
model = models[arch]

#Turn off the parameters training
for param in model.parameters():
    param.requires_grad = False

#Define a new Classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_output[0])),
                          ('relu', nn.ReLU()),
                          ('drop', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_output[0], hidden_output[1])),
                          ('output', nn.LogSoftmax(dim=1))
                           ]))
                    
#Replace the classifier from the VGG model with our classifier
model.classifier = classifier

# Move input and label tensors to the GPU, if GPU is available
model.to(device)

print("training model...")

#Defining Loss function and Optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
                                   
#Implement a function for the validation pass
def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(validloader):
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy                                  
                                 
#TEST NEW MODEL
print_every = 10
steps = 0
running_loss = 0

for e in range(epochs):
    model.train()
    for ii, (inputs, labels) in enumerate(trainloader):
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        steps += 1
         
        #Making the gradients zero to aviod gradient accumulation    
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
              
        #Checking the model performance on validation set
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
        running_loss = 0
            
        #Training is back on
        model.train()  

model.class_to_idx = train_datasets.class_to_idx

# Generating and saving the checkpoint 

checkpoint = {'classifier':model.classifier,
              'arch':arch,
              'learning_rate': learning_rate,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}

checkpoint_path = "{}/{}".format(in_arg.save_dir, 'checkpoint.pth')

torch.save(checkpoint, checkpoint_path)

                                   
                                   
    
