from logging import exception
import torch
from torch import Size, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, Resize

def Datasets(Name):
    if Name == "MNIST":
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

    elif Name == "CIFAR10":
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=Compose([Resize(28),ToTensor()])
        )

        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=Compose([Resize(28),ToTensor()])
        )

    else:
        raise exception("------No Such Dataset------")
    
    return training_data, test_data