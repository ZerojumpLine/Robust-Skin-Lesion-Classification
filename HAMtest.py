#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


# In[ ]:


# load the model
from models.resnet_skin import network
# default settings
num_classes = 7
model = network(name = 'resnet50', num_classes = num_classes)


# In[ ]:


# load the model parameters
model_ckpt = './checkpoints_cls/HAM_CE_None_0_cosine/ckpt.best.pth.tar'
gpu = 0
model = model.cuda(gpu)
checkpoint = torch.load(model_ckpt, map_location = 'cuda:' + str(gpu))


# In[ ]:


# have some naming issues
# in checkpoint, parameters are 'module.encoder.conv1.weight'
# in model, parameters are 'encoder.conv1.weight'


# In[ ]:


# this is not very elegant, but should work.
ckpt_toload = dict()
for name in checkpoint['state_dict']:
    name_model = name[7:]
    ckpt_toload[name_model] = checkpoint['state_dict'][name]
model.load_state_dict(ckpt_toload)


# In[ ]:


# prepare the test data.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ResizeTest = transforms.Resize(256)

transform_val = transforms.Compose([
        ResizeTest,
        transforms.ToTensor(),
        normalize,
    ])

data_folder = './skinlesiondatasets/SkinLesionDatasets/'
train_dataset = datasets.ImageFolder(root=os.path.join(data_folder, 'HAMtrain'), transform=transform_val)
#
val_dataset = datasets.ImageFolder(root=os.path.join(data_folder, 'HAMval'), transform=transform_val)
val_dataset.class_to_idx = train_dataset.class_to_idx
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)
#
test_dataset = datasets.ImageFolder(root=os.path.join(data_folder, 'HAMtest'), transform=transform_val)
test_dataset.class_to_idx = train_dataset.class_to_idx
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)


# In[ ]:


cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]


# In[ ]:


cifarresultsdir = './skinresults/'
model.eval()
logits = []
preds = []
targets = []
for i, (input, target) in enumerate(tqdm(val_loader)):
    input = input.cuda(gpu, non_blocking = True)
    logits_test = model(input)
    preds_test = F.softmax(logits_test, dim = 1)
    targets_test = F.one_hot(target, num_classes = num_classes)
    logits.append(logits_test.cpu().detach())
    preds.append(preds_test.cpu().detach())
    targets.append(targets_test)
    
logits = torch.cat(logits, dim=0)
preds = torch.cat(preds, dim=0)
targets = torch.cat(targets, dim=0)
    
df = pd.DataFrame(data=preds.numpy(), columns=cols_names_classes)
df_logits = pd.DataFrame(data=logits.numpy(), columns=cols_names_logits)
df_targets = pd.DataFrame(data=targets.numpy(), columns=cols_names_targets)
df = pd.concat([df, df_logits, df_targets], axis=1)
df.to_csv(os.path.join(cifarresultsdir, 'predictions_val.csv'), index=False)


# In[ ]:


# Other datasets here
datasetlist = ['BCN', 'D7P', 'MSK', 'OTH', 'PH2', 'UDA', 'VIE']
model.eval()
cifarresultsdir = './skinresults/'
for datasetn in datasetlist:
    
    print('Processing dataset ' + datasetn)
    
    csvsavename = 'predictions_test_' + datasetn +  '.csv'
    
    val_dataset = datasets.ImageFolder(root=os.path.join(data_folder, datasetn), transform=transform_val)

    val_dataset.class_to_idx = train_dataset.class_to_idx
    
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True)
    
    logits = []
    preds = []
    targets = []
    for i, (input, target) in enumerate(tqdm(val_loader)):
        input = input.cuda(gpu, non_blocking = True)
        logits_test = model(input)
        preds_test = F.softmax(logits_test, dim = 1)
        targets_test = F.one_hot(target, num_classes = num_classes)
        logits.append(logits_test.cpu().detach())
        preds.append(preds_test.cpu().detach())
        targets.append(targets_test)

    logits = torch.cat(logits, dim=0)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    df = pd.DataFrame(data=preds.numpy(), columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits.numpy(), columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets.numpy(), columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(cifarresultsdir, csvsavename), index=False)
    


# In[ ]:


# corrupted datasets here
data_folder = './skinlesiondatasets/SkinLesionDatasets_C/'

datasetlist = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 
               'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 
               'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 
               'snow', 'spatter', 'speckle_noise', 'zoom_blur']
severitylist = [1, 2, 3, 4, 5]

model.eval()
for datasetn in datasetlist:
    
    for severityindex in severitylist:
    
        print('Processing dataset ' + datasetn + ' severity ' + str(severityindex))

        csvsavename = 'predictions_test_' + datasetn + '_' + str(severityindex) + '.csv'

        val_dataset = datasets.ImageFolder(root=os.path.join(data_folder, datasetn, str(severityindex)), transform=transform_val)
        
        val_dataset.class_to_idx = train_dataset.class_to_idx

        val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True)

        logits = []
        preds = []
        targets = []
        for i, (input, target) in enumerate(tqdm(val_loader)):
            input = input.cuda(gpu, non_blocking = True)
            logits_test = model(input)
            preds_test = F.softmax(logits_test, dim = 1)
            targets_test = F.one_hot(target, num_classes = num_classes)
            logits.append(logits_test.cpu().detach())
            preds.append(preds_test.cpu().detach())
            targets.append(targets_test)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        df = pd.DataFrame(data=preds.numpy(), columns=cols_names_classes)
        df_logits = pd.DataFrame(data=logits.numpy(), columns=cols_names_logits)
        df_targets = pd.DataFrame(data=targets.numpy(), columns=cols_names_targets)
        df = pd.concat([df, df_logits, df_targets], axis=1)
        df.to_csv(os.path.join(cifarresultsdir, csvsavename), index=False)
    


# In[ ]:





# In[ ]:




