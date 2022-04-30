import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Resize
import os
import numpy as np

def Model_Selector(Name):
    print("Model Name: {}".format(Name))
    if Name == "CNN":
        return CNN()
    elif Name == "ViT":
        return ViT()
    elif Name == "FC":
        return FC()
    else:
        raise Exception("------No Such Model------")

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(num_features=28 * 28, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        if x.size()[1] > 1:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[: ,2:3, :, :]
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential( # (1, 28, 28)
            nn.Conv2d(1, 3, (7, 7), (1, 1), (1, 1)), # (3, 24, 24)
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)) , # (3, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True),
            nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)), # (3, 12, 12)
            nn.MaxPool2d((2,2) , (2,2) , (0, 0) ), # (3, 6, 6)
            nn.Flatten(), # (3*6*6)
            nn.Linear(3*6*6, 128), 
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(128, 64, bias = True),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(64, 32, bias = True),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True),
            nn.Linear(32, 10, bias = True),
        )

    def forward(self, x):
        if x.size()[1] > 1:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[: ,2:3, :, :]
        logits = self.cnn_model(x)
        return logits


class ViT(nn.Module):
    def __init__(self): # (4 * 4) * 7 * 7
        super(ViT, self).__init__()

        self.pos_emb = torch.rand((7, 7, 16))

        self.to_Q1 = [nn.Linear(32, 16, bias=False) for _ in range(49)]
        self.to_K1 = [nn.Linear(32, 16, bias=False) for _ in range(49)]
        self.to_V1 = [nn.Linear(32, 16, bias=False) for _ in range(49)]
        self.softmax1 = nn.Softmax()
        self.norm1 = nn.BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.to_Q2 = [nn.Linear(16, 8, bias=False) for _ in range(49)]
        self.to_K2 = [nn.Linear(16, 8, bias=False) for _ in range(49)]
        self.to_V2 = [nn.Linear(16, 8, bias=False) for _ in range(49)]
        self.softmax2 = nn.Softmax()
        self.norm2 = nn.BatchNorm1d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.FC = nn.Sequential(
            nn.Linear(49 * 8, 16, True),
            nn.ReLU(),
            nn.Linear(16, 10, True),
            nn.Softmax()
        )

    def self_attention(self, Q, K, V):
        dk = Q.size()[1]
        token = []
        batch_size = Q.size()[0]
        for i in range(batch_size):
            token.append( (Q[i].T @ K[i] / (dk)**(1/2)) @ V[i].T )
        return torch.stack(token).permute(0, 2, 1) # (49, 16)

    def forward(self, x):
        if x.size()[1] > 1:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[: ,2:3, :, :] # (1, 28, 28)
        batch_size = x.size()[0]
        token1 = []
        for i in range(7):
            for j in range(7):
                token1.append(torch.cat((x[:, 0, 4*i: 4*(i+1), 4*j : 4*(j+1)].reshape((batch_size, 16)),
                            torch.repeat_interleave(self.pos_emb[i, j].reshape((1,16)), repeats = batch_size, dim = 0)), dim=1))

        token1 = torch.stack(token1).permute(1,0,2) # (49, 32)

        Q1 = (torch.stack([self.to_Q1[i](token1[:, i, :]) for i in range(49)])).permute(1,0,2) # (49, 16)
        K1 = (torch.stack([self.to_K1[i](token1[:, i, :]) for i in range(49)])).permute(1,0,2) # (49, 16)
        V1 = (torch.stack([self.to_V1[i](token1[:, i, :]) for i in range(49)])).permute(1,0,2) # (49, 16)

        token2 = self.softmax1(self.self_attention(Q1, K1, V1)) # (49, 16)

        Q2 = (torch.stack([self.to_Q2[i](token2[:, i, :]) for i in range(49)])).permute(1,0,2) # (49, 8)
        K2 = (torch.stack([self.to_K2[i](token2[:, i, :]) for i in range(49)])).permute(1,0,2) # (49, 8)
        V2 = (torch.stack([self.to_V2[i](token2[:, i, :]) for i in range(49)])).permute(1,0,2) # (49, 8)

        token3 = self.softmax2(self.self_attention(Q2, K2, V2)) # (49, 8)

        flat = token3.reshape((batch_size, 49 * 8)) # (49 * 8)

        logits = self.FC(flat) # (10)

        return logits