import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from tqdm import tqdm
# import wandb
import argparse
from torch.utils.tensorboard import SummaryWriter

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        """
        TODO 2.1.1 : Fill in self.convs following the given architecture 
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """
        self.convs = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, 3, 2, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 128, 3, 2, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 256, 3, 2, 1))
        self.conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256

        self.fc = nn.Linear(self.conv_out_dim, self.latent_dim)
        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.latent_dim, num_classes))

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.view(x.shape[0], -1))
        x = self.classifier(x)

        return x

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        self.conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256
        self.fc = nn.Linear(self.conv_out_dim, 2 * latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.view(x.shape[0], -1))
        return (x[:, :self.latent_dim], x[:, self.latent_dim: ])

def load_model(model, load_path):
    pretrained = torch.load(load_path)
    model_state = model.state_dict()

    for name in pretrained:
        if name not in model_state:
            continue

        param = pretrained[name]

        model_state[name].copy_(param)

def train_step(model, train_dataloader, optimizer, epoch, writer):
    loss_l = []
    pbar = tqdm(train_dataloader)
    # for imgs, target in train_dataloader:
    for imgs, target in pbar:
        pred = model(imgs.cuda())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, target.cuda())
        loss_l.append(loss.item())
        pbar.set_description(f'Epoch={epoch}, Loss={np.mean(loss_l)}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # wandb.log({'train_loss': np.mean(loss_l), 'train_step': epoch})
    writer.add_scalar('Loss/Train', np.mean(loss_l), epoch)

def validate(model, test_loader, epoch, writer):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.detach().cpu() == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    # wandb.log({'val_acc': np.mean(100 * correct // total), 'train_step': epoch})
    writer.add_scalar('Val_acc', np.mean(100 * correct // total), epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BYoL Classification')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--log_name', type=str)
    parser.add_argument('--num_epochs', type=int)
    args = parser.parse_args()

    writer = SummaryWriter(args.log_name)
    model = Encoder((3, 32, 32), 128, 10)
    model.cuda()
    load_model(model, args.model_path)

    # wandb.init(project="16_824_project", entity = "ayushpandey34", name = 'train_byol_orig', reinit=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = args.num_epochs
    train_dataloader, test_dataloader = get_dataloaders(batch_size = 32)
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        train_step(model, train_dataloader, optimizer, epoch, writer)
        if epoch % 2 == 0:
            model.eval()
            validate(model, test_dataloader, epoch, writer)
            model.train()
