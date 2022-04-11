from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import time
import os


def avg_dict(all_metrics):
    keys = all_metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = np.mean([all_metrics[i][key].cpu().detach().numpy() for i in range(len(all_metrics))])
    return avg_metrics

def save_samples(samples, fname, nrow=6, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255.).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    f = plt.figure()
    f.clear()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.tight_layout()
    plt.savefig(fname)


def get_dataloaders(train_set, test_set, batch_size = 256):    
    train_loader = torch.utils.data.DataLoader(train_set,batch_size= batch_size, shuffle=True,num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(test_set,batch_size= batch_size, shuffle=False,num_workers=4, pin_memory=True)
    return train_loader, val_loader

def vis_recons(model, x, z, _file):

    with torch.no_grad():
        x_recon = torch.clamp(model(z), -1, 1)
    
    reconstructions = torch.stack((x, x_recon), dim=1).view(-1, 3, 32, 32) * 0.5 + 0.5  
    reconstructions = reconstructions.permute(0, 2, 3, 1).cpu().numpy() * 255

    save_samples(reconstructions, _file+'_recons.png')