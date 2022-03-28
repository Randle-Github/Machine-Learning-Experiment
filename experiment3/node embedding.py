import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import DeepWalk as dp
import torch
import torch.nn as nn


class NeuralNetworks(nn.Module):
    def __init__(self, one_hot_len, embedding_len):
        super(NeuralNetworks, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(one_hot_len, embedding_len),
            nn.Linear(embedding_len, one_hot_len)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def loss_func(self, Logits, target):
        self.mse_loss = self.loss_fn(Logits, target)
        return self.mse_loss

    def backward_func(self):
        self.mse_loss.backward()


one_hot_len = 821
embedding_len = 2
model = NeuralNetworks(one_hot_len, embedding_len)
epoches = 100
learning_rate = 5e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
batch_size = 64


def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    aver_loss = torch.tensor([0.])
    for batch, (X, y) in enumerate(dataloader):
        pred = model.forward(X)
        loss = model.loss_func(pred, y)
        optimizer.zero_grad()
        model.backward_func()
        optimizer.step()
        temp = torch.tensor(loss.item())
        aver_loss += temp
    print("average loss:", aver_loss / size)


input = torch.eye(821, 821)
output = dp.main()

training_dataset = TensorDataset(input, output)
training_data = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

for i in range(epoches):
    print(f"Epoch {i + 1}\n------------")
    train_loop(training_data, model, optimizer)
    print("")
optimizer = torch.optim.SGD(model.parameters(), lr=0)
train_loop(training_data, model, optimizer)

file = open("word_embedding_visual.txt", "w")
for vec in model.model.parameters():
    vec = vec.tolist()
    for i in range(821):
        for j in range(embedding_len):
            print(vec[j][i], end=" ", file=file)
        print("", file=file)
    break
