import torch
import torchvision
from torch.utils.data import DataLoader


def torch_version():
    return torch.__version__


def torchvision_version():
    return torchvision.__version__


class CIFAR10DataLoader:
    def __init__(self):
        pass

    def load(self, root, transform, batch_size, num_workers):
        train_set, trainloader, test_set, testloader = None, None, None, None
        try:
            # trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
            # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
            #                                           num_workers=num_workers)
            #
            # testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
            # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

            trainloader = DataLoader(train_set, batch_size=10, shuffle=True)
            testloader = DataLoader(test_set, batch_size=10, shuffle=False)

        except Exception as e:
            print("exception in data loading")
            print(e)
        # return trainset, trainloader, testset, testloader
        return train_set, trainloader, test_set, testloader
