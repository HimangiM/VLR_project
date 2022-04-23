import torch
from byol_pytorch import BYOL
from torchvision import models
import torch.nn as nn
from utils import *
from custom_datasets import Trainset, Testset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
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

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.view(x.shape[0], -1))

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


def training_loop(learner, optimizer, train_dataset, dict_positive, train_dataloader, epoch, writer):
    loss_l = []
    pbar = tqdm(train_dataloader)

    for x, target, idxs in tqdm(train_dataloader):
        positive_imgs = []
        for idx in idxs:
            positive_imgs.append(train_dataset.__getitem__(dict_positive[idx.item()])[0])

        positive_imgs = torch.stack(positive_imgs, dim = 0)
        loss = learner((x, positive_imgs))
        loss_l.append(loss.item())
        writer.add_scalar('Loss/Train', np.mean(loss_l), epoch)
        pbar.set_description(f'Epoch={epoch}, Loss={np.mean(loss_l)}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learner.update_moving_average()

if __name__ == '__main__':
    writer = SummaryWriter()
    encoder_net = Encoder((3, 32, 32), 128)

    train_dataset = Trainset()
    test_dataset = Testset()
    train_dataloader, test_data_loader = get_custom_dataloaders(train_dataset, test_dataset, batch_size=32)

    learner = BYOL(
        encoder_net,
        image_size = 32,
        hidden_layer = 'fc'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    # def sample_unlabelled_images():
    #     return torch.randn(20, 3, 32, 32)
    num_epochs = 20
    import pickle5 as pkl
    dict_positive = pkl.load(open('nearest_neighbor.pickle', 'rb'))
    for epoch in range(num_epochs):
        training_loop(learner, opt, train_dataset, dict_positive, train_dataloader, epoch, writer)

    # save your improved network
    torch.save(encoder_net.state_dict(), './improved-net.pt')


# Baseline: