# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

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
from get_predict_args import get_predict_args
#from get_input_args import get_input_args

# Function that checks command line arguments using in_arg  
pred_arg = get_predict_args()
#in_args = get_input_args()
print(pred_arg)


#label mapping
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


#Building and training network

#Defining Hyperparameters
top_k = pred_arg.top_k
checkpoint_path = pred_arg.checkpoint
image_path = pred_arg.image_path
gpu = pred_arg.gpu

arch = 'vgg16' #need to figure out


if pred_arg.gpu == True: 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else: device = torch.device("cpu")
    
#TODO: Rebuilding Model
def rebuild_vgg(filepath):
    checkpoint = torch.load(filepath)
    
    
    if checkpoint['arch'] == 'vgg13':
        model= models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'vgg16':      
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Sorry base architecture note recognized")
            
    #Create Classifier
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       
    return optimizer, model
    
optimizer, model = rebuild_vgg(checkpoint_path)
print(model)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    #Get the image and image dimentions
    img = Image.open(image_path)
    width, height = img.size
    
    #Resize image keeping aspect ratio to shortest size (256)
    if width < height:
        new_height = int(height/width)
        resized_image = img.resize((256, new_heigh))
    else:
        new_width = int(256*(width/height))
        resized_image = img.resize((new_width,256))
        
    width, height = resized_image.size
    print('Resized image', width, height)#to check size
    
    #Croping the image to 224x224
    crop_width = 224
    crop_height = 224
    left = (width - crop_width)/2
    top = (height - crop_height)/2
    right = (width - crop_width)/2 + crop_width
    bottom = (height - crop_height)/2 + crop_height
    resized_image = resized_image.crop((left, top, right, bottom))
    width, height = resized_image.size
    print('croped dimentions',width,height) #to verify size
    
    #Turn image into array
    array_image = np.array(resized_image)
    
    #Turns all RGB color values into range(0:1)
    array_image = array_image/255
    
    # normalize and transpose images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    array_image = (array_image - mean) / std
    array_image = array_image.transpose((2, 0, 1))
    
    # modify the output of process_image function to a tensor
    processed_image = torch.from_numpy(array_image)
    tensor_image = processed_image.float()
    print(type(tensor_image))
  
    return tensor_image 

process_image(image_path)


def predict(image_path, model, top_k, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print('Predicting image class...')
       
    model.eval()
    model.to('cuda')
   
    #Process Image
    img = Image.open(image_path)
    img = process_image(image_path)
    model_input = img.float().cuda()

    #do a fwd pass
    output = model.forward(model_input.unsqueeze(0))
    
    #Calculate Probs
    probs = torch.exp(output)
    
    #Top Probs
    top_probs = probs.topk(top_k)
    prob = top_probs[0].cpu().detach().numpy().tolist()[0]
    idx = top_probs[1].cpu().detach().numpy().tolist()[0]
    
    '''prob = top_probs.cpu().detach().numpy().tolist()[0]
    idx = top_probs.cpu().detach().numpy().tolist()[0]'''
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}     
    classes = [model.idx_to_class[x] for x in idx]

    
    return prob, classes
    
    
probs, flowers = predict(image_path, model, top_k)
print(probs)
print(flowers)
