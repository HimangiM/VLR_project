from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from model import AEModel
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import time
import os
from utils import *
import wandb
from custom_datasets import Trainset, Testset
import pickle
import argparse

def ae_loss(model, x):
    pred, _ = model(x)
    loss = torch.sum(torch.square(x - pred)) / x.shape[0]
    
    return loss, OrderedDict(recon_loss=loss)

def vae_loss(model, x, beta = 1):
    pred, latent_variable = model(x)
    recon_loss = torch.sum(torch.square(pred - x)) / x.shape[0]
    mew = latent_variable[0]
    log_var = latent_variable[1]

    kl_loss = 0.5 * torch.sum(torch.square(mew) + torch.exp(log_var) - log_var - 1) / x.shape[0]

    total_loss = recon_loss + beta*kl_loss
    return total_loss, OrderedDict(recon_loss=recon_loss, kl_loss=kl_loss)


def constant_beta_scheduler(target_val = 1):
    def _helper(epoch):
        return target_val
    return _helper

def linear_beta_scheduler(max_epochs=None, target_val = 1):
    def _helper(epoch):
        return (epoch * target_val) / max_epochs

    return _helper

def run_train_epoch(model, loss_mode, train_loader, optimizer, beta = 1, grad_clip = 1):
    model.train()
    all_metrics = []
    for x, _ in train_loader:
        x = preprocess_data(x)
        if loss_mode == 'ae':
            loss, _metric = ae_loss(model, x)
        elif loss_mode == 'vae':
            loss, _metric = vae_loss(model, x, beta)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return avg_dict(all_metrics)


def get_val_metrics(model, loss_mode, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = preprocess_data(x)
            if loss_mode == 'ae':
                _, _metric = ae_loss(model, x)
            elif loss_mode == 'vae':
                _, _metric = vae_loss(model, x)
            all_metrics.append(_metric)

    return avg_dict(all_metrics)

def main(log_dir, loss_mode = 'vae', beta_mode = 'constant', num_epochs = 20, batch_size = 256, latent_size = 256,
         target_beta_val = 1, grad_clip=1, lr = 1e-3, eval_interval = 5):

    print(loss_mode, latent_size)
    os.makedirs('data/'+ log_dir, exist_ok = True)
    train_loader, val_loader = get_dataloaders()

    variational = True if loss_mode == 'vae' else False
    model = AEModel(variational, latent_size, input_shape = (3, 32, 32)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    vis_x = next(iter(val_loader))[0][:36]
    
    if beta_mode == 'constant':
        beta_fn = constant_beta_scheduler(target_val = target_beta_val) 
    elif beta_mode == 'linear':
        beta_fn = linear_beta_scheduler(max_epochs=num_epochs, target_val = target_beta_val) 

    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, loss_mode, train_loader, optimizer, beta_fn(epoch))
        val_metrics = get_val_metrics(model, loss_mode, val_loader)
        if 'kl_loss' in train_metrics:
            wandb.log({'train_recon_loss': train_metrics['recon_loss'],
                       'val_recon_loss': val_metrics['recon_loss'],
                       'train_kl_loss': train_metrics['kl_loss'],
                       'val_kl_loss': val_metrics['kl_loss'],
                       'run_epoch': epoch})
        else:
            wandb.log({'train_recon_loss': train_metrics['recon_loss'],
                       'val_recon_loss': val_metrics['recon_loss'],
                       'run_epoch': epoch})

        if (epoch+1)%eval_interval == 0:
            print(epoch, train_metrics)
            print(epoch, val_metrics)

            vis_recons(model, vis_x, 'data/'+log_dir+ '/epoch_'+str(epoch))
            if loss_mode == 'vae':
                vis_samples(model, 'data/'+log_dir+ '/epoch_'+str(epoch))
    
        if epoch == num_epochs - 1:
            model.eval()
            train_set = Trainset()
            latent_vectors = {}
            for i in range(len(train_set)):
                img, _, _ = train_set[i]
                z = get_latent_vectors(model, img.unsqueeze(0)).detach().cpu().numpy()[0]
                latent_vectors[i] = z

            with open(f'latent_vectors_{loss_mode}.pickle', 'wb') as file:
                pickle.dump(latent_vectors, file, protocol=pickle.HIGHEST_PROTOCOL)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('latent_dim', type = int)
    args = parser.parse_args()

    latent_dim = args.latent_dim  
    wandb.init(project="vlr_project", entity = "ayushpandey34", name = args.mode, reinit=True)
    if args.mode == 'ae':
        main('ae_latent' + str(latent_dim), loss_mode = 'ae',  num_epochs = 100, latent_size = latent_dim)
    elif args.mode == 'vae':
        main('vae_latent' + str(latent_dim), loss_mode = 'vae', num_epochs = 100, latent_size = latent_dim)

