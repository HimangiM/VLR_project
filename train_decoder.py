
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from collections import OrderedDict
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from utils import *
import argparse
from custom_datasets import Trainset, Testset



class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        #TODO 2.1.1: fill in self.base_size
        self.base_size = 256*4*4
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        
        """
        TODO 2.1.1 : Fill in self.deconvs following the given architecture 
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, z):
        #TODO 2.1.1: forward pass through the network, first through self.fc, then self.deconvs.
        x = self.fc(z)
        x = x.view(-1, 256,4,4)
        x = self.deconvs(x)
        return x






def run_train_epoch(model, Z, train_loader, optimizer):
    model.train()
    all_metrics = []
    for x, _ , idx in train_loader:
        x = x.cuda()
        z_idx = Z[idx].cuda()
        z_inp = torch.nn.Parameter(z_idx, requires_grad=True).cuda()
        optimizer.add_param_group({"params": z_inp})
        pred = model(z_inp)
        loss = nn.MSELoss(reduction = 'sum')(pred, x)  / x.shape[0]
        _metric = OrderedDict(recon_loss=loss)
        all_metrics.append(_metric)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return avg_dict(all_metrics)




def main(log_dir,  num_epochs = 20, batch_size = 256, latent_size = 256, lr = 1e-3, eval_interval = 5):
    train_set = Trainset()
    test_set = Testset()
    len_train_set = train_set.__len__()
    mean = torch.ones(len_train_set, latent_size)
    std = torch.ones(len_train_set, latent_size)
    Z = torch.normal(mean= mean, std= std).cuda()
    model = Decoder(latent_size).cuda()
    train_loader, val_loader = get_dataloaders(train_set, test_set, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    #vis_x,_, vis_idx = next(iter(train_loader))[:36]
    for epoch in range(num_epochs):
        print('epoch', epoch)
        train_metrics = run_train_epoch(model, Z, train_loader, optimizer)
        if (epoch+1)%eval_interval == 0:
            print(epoch, train_metrics)
            #vis_recons(model, vis_x, vis_idx, log_dir+ '/epoch_'+str(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--latent_size', type=int, default=128)
    parser.add_argument('--eval_interval', type=int, default= 1)
    parser.add_argument('--batch_size', type=int, default=256, help='The number of images in a batch.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (default 0.001)')

    parser.add_argument('--log_dir', type=str, default="exp", help='The name of the log dir')
    args = parser.parse_args()

    main(args.log_dir,  num_epochs = args.num_epochs,  batch_size = args.batch_size, latent_size = args.latent_size, lr = args.lr, eval_interval = args.eval_interval)