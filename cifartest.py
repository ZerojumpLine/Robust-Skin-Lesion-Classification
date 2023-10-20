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
import models
# default settings
arch = 'resnet32'
num_classes = 10
use_norm = False
model = models.__dict__[arch](num_classes=num_classes, use_norm=use_norm)

cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

# prepare the test data.
transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)


# In[ ]:


import random
random.seed(79)
test_ind = list(range(10000))
random.shuffle(test_ind)


# In[ ]:


val_ind = test_ind[:3000]
testc_indx_1 = test_ind[3000:]
testc_indx_2 = [x+10000 for x in test_ind[3000:]]
testc_indx_3 = [x+20000 for x in test_ind[3000:]]
testc_indx_4 = [x+30000 for x in test_ind[3000:]]
testc_indx_5 = [x+40000 for x in test_ind[3000:]]
testc_indx = testc_indx_1 + testc_indx_2 + testc_indx_3 + testc_indx_4 + testc_indx_5


# In[ ]:


val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, sampler = torch.utils.data.SubsetRandomSampler(val_ind))


# In[ ]:


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index])
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)


# # CIFAR-10

# In[ ]:


# load the model parameters
# vanilla training
model_ckpt = './checkpoints_cls/cifar10_resnet32_CE_None_exp_0.01_0/ckpt.pth.tar'
# ci_1
model_ckpt_ci1 = './checkpoints_cls/cifar10_resnet32_LDAM_None_exp_0.01_0/ckpt.pth.tar'
# ci_2
model_ckpt_ci2 = './checkpoints_cls/cifar10_resnet32_CE_DRW_exp_0.01_0/ckpt.pth.tar'
# rl_1
model_ckpt_rl1 = './checkpoints_cls/cifar10_resnet32_CE_None_exp_0.01_0_cutout/ckpt.pth.tar'
# rl_2
model_ckpt_rl2 = './checkpoints_cls/cifar10_resnet32_CE_None_exp_0.01_0_randaugment/ckpt.pth.tar'


# ## CIFAR-10, baseline

# In[ ]:


gpu = 0
model = model.cuda(gpu)
checkpoint = torch.load(model_ckpt, map_location = 'cuda:' + str(gpu))
model.load_state_dict(checkpoint['state_dict'])


# In[ ]:


cifarresultsdir = './cifar10results/baseline/'
if not os.path.exists(cifarresultsdir):
    os.makedirs(cifarresultsdir)

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


cifarcdatadir = './CIFAR-10-C/'

alltestfiles = os.listdir(cifarcdatadir)
alltestfiles.remove('labels.npy')

labelcifar = np.load(cifarcdatadir + 'labels.npy')

for testc in tqdm(alltestfiles):
    
    print('Processing corruption: ', testc[:-4])
    
    datasetexample = np.load(cifarcdatadir + testc)
    
    val_datasetc = MyDataset(datasetexample, labelcifar, transform=transform_val)
    val_loaderc = torch.utils.data.DataLoader(val_datasetc, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, sampler = torch.utils.data.SubsetRandomSampler(testc_indx))
    
    savefilename = 'predictions_val_' + testc[:-4] + '.csv'
    
    model.eval()
    logits = []
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loaderc):
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
    df.to_csv(os.path.join(cifarresultsdir, savefilename), index=False)
    


# ## CIFAR-10, ci1

# In[ ]:


gpu = 0
use_norm = True
model = models.__dict__[arch](num_classes=num_classes, use_norm=use_norm)
model = model.cuda(gpu)
checkpoint = torch.load(model_ckpt_ci1, map_location = 'cuda:' + str(gpu))
model.load_state_dict(checkpoint['state_dict'])


# In[ ]:


cifarresultsdir = './cifar10results/ci1/'
if not os.path.exists(cifarresultsdir):
    os.makedirs(cifarresultsdir)

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


cifarcdatadir = './CIFAR-10-C/'

alltestfiles = os.listdir(cifarcdatadir)
alltestfiles.remove('labels.npy')

labelcifar = np.load(cifarcdatadir + 'labels.npy')

for testc in tqdm(alltestfiles):
    
    print('Processing corruption: ', testc[:-4])
    
    datasetexample = np.load(cifarcdatadir + testc)
    
    val_datasetc = MyDataset(datasetexample, labelcifar, transform=transform_val)
    val_loaderc = torch.utils.data.DataLoader(val_datasetc, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, sampler = torch.utils.data.SubsetRandomSampler(testc_indx))
    
    savefilename = 'predictions_val_' + testc[:-4] + '.csv'
    
    model.eval()
    logits = []
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loaderc):
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
    df.to_csv(os.path.join(cifarresultsdir, savefilename), index=False)
    


# ## CIFAR-10, ci2

# In[ ]:


gpu = 0
model = model.cuda(gpu)
checkpoint = torch.load(model_ckpt_ci2, map_location = 'cuda:' + str(gpu))
model.load_state_dict(checkpoint['state_dict'])


# In[ ]:


cifarresultsdir = './cifar10results/ci2/'
if not os.path.exists(cifarresultsdir):
    os.makedirs(cifarresultsdir)

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


cifarcdatadir = './CIFAR-10-C/'

alltestfiles = os.listdir(cifarcdatadir)
alltestfiles.remove('labels.npy')

labelcifar = np.load(cifarcdatadir + 'labels.npy')

for testc in tqdm(alltestfiles):
    
    print('Processing corruption: ', testc[:-4])
    
    datasetexample = np.load(cifarcdatadir + testc)
    
    val_datasetc = MyDataset(datasetexample, labelcifar, transform=transform_val)
    val_loaderc = torch.utils.data.DataLoader(val_datasetc, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, sampler = torch.utils.data.SubsetRandomSampler(testc_indx))
    
    savefilename = 'predictions_val_' + testc[:-4] + '.csv'
    
    model.eval()
    logits = []
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loaderc):
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
    df.to_csv(os.path.join(cifarresultsdir, savefilename), index=False)
    


# ## CIFAR-10, rl1

# In[ ]:


gpu = 0
model = model.cuda(gpu)
checkpoint = torch.load(model_ckpt_rl1, map_location = 'cuda:' + str(gpu))
model.load_state_dict(checkpoint['state_dict'])


# In[ ]:


cifarresultsdir = './cifar10results/rl1/'
if not os.path.exists(cifarresultsdir):
    os.makedirs(cifarresultsdir)

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


cifarcdatadir = './CIFAR-10-C/'

alltestfiles = os.listdir(cifarcdatadir)
alltestfiles.remove('labels.npy')

labelcifar = np.load(cifarcdatadir + 'labels.npy')

for testc in tqdm(alltestfiles):
    
    print('Processing corruption: ', testc[:-4])
    
    datasetexample = np.load(cifarcdatadir + testc)
    
    val_datasetc = MyDataset(datasetexample, labelcifar, transform=transform_val)
    val_loaderc = torch.utils.data.DataLoader(val_datasetc, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, sampler = torch.utils.data.SubsetRandomSampler(testc_indx))
    
    savefilename = 'predictions_val_' + testc[:-4] + '.csv'
    
    model.eval()
    logits = []
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loaderc):
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
    df.to_csv(os.path.join(cifarresultsdir, savefilename), index=False)
    


# ## CIFAR-10, rl2

# In[ ]:


gpu = 0
model = model.cuda(gpu)
checkpoint = torch.load(model_ckpt_rl2, map_location = 'cuda:' + str(gpu))
model.load_state_dict(checkpoint['state_dict'])


# In[ ]:


cifarresultsdir = './cifar10results/rl2/'
if not os.path.exists(cifarresultsdir):
    os.makedirs(cifarresultsdir)

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


cifarcdatadir = './CIFAR-10-C/'

alltestfiles = os.listdir(cifarcdatadir)
alltestfiles.remove('labels.npy')

labelcifar = np.load(cifarcdatadir + 'labels.npy')

for testc in tqdm(alltestfiles):
    
    print('Processing corruption: ', testc[:-4])
    
    datasetexample = np.load(cifarcdatadir + testc)
    
    val_datasetc = MyDataset(datasetexample, labelcifar, transform=transform_val)
    val_loaderc = torch.utils.data.DataLoader(val_datasetc, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, sampler = torch.utils.data.SubsetRandomSampler(testc_indx))
    
    savefilename = 'predictions_val_' + testc[:-4] + '.csv'
    
    model.eval()
    logits = []
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loaderc):
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
    df.to_csv(os.path.join(cifarresultsdir, savefilename), index=False)
    


# # CIFAR-100

# In[ ]:


# load the model parameters
# vanilla training
model_ckpt = './checkpoints_cls/cifar100_resnet32_CE_None_exp_0.01_0/ckpt.pth.tar'


# In[ ]:


# load the model
import models
# default settings
arch = 'resnet32'
num_classes = 100
use_norm = False
model = models.__dict__[arch](num_classes=num_classes, use_norm=use_norm)

cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

# prepare the test data.
transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)


# In[ ]:


gpu = 0
model = model.cuda(gpu)
checkpoint = torch.load(model_ckpt, map_location = 'cuda:' + str(gpu))
model.load_state_dict(checkpoint['state_dict'])


# In[ ]:


cifarresultsdir = './cifar100results/baseline/'
if not os.path.exists(cifarresultsdir):
    os.makedirs(cifarresultsdir)

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


cifarcdatadir = './CIFAR-100-C/'

alltestfiles = os.listdir(cifarcdatadir)
alltestfiles.remove('labels.npy')

labelcifar = np.load(cifarcdatadir + 'labels.npy')

for testc in tqdm(alltestfiles):
    
    print('Processing corruption: ', testc[:-4])
    
    datasetexample = np.load(cifarcdatadir + testc)
    
    val_datasetc = MyDataset(datasetexample, labelcifar, transform=transform_val)
    val_loaderc = torch.utils.data.DataLoader(val_datasetc, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, sampler = torch.utils.data.SubsetRandomSampler(testc_indx))
    
    savefilename = 'predictions_val_' + testc[:-4] + '.csv'
    
    model.eval()
    logits = []
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loaderc):
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
    df.to_csv(os.path.join(cifarresultsdir, savefilename), index=False)
    


# In[ ]:




