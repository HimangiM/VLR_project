from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class Trainset(Dataset):
    def __init__(self, path = './'):
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=train_transforms)
                            
    def __getitem__(self, index):
        data, target = self.cifar10[index]
                                                            
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

class Testset(Dataset):
    def __init__(self, path = './'):
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                                      std=[0.229, 0.224, 0.225])])
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=False,
                                        transform=test_transforms)
                            
    def __getitem__(self, index):
        data, target = self.cifar10[index]
                                                            
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

if __name__ == '__main__':
    train_set = Trainset()
    test_set = Testset()
    print(train_set[10])
    print(test_set[10])

