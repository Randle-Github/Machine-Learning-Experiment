import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import DeepWalk as dp
import torch
import torch.nn as nn


class NeuralNetworks(nn.Module):
    def __init__(self):
        super(NeuralNetworks, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x, x0, x1):  # negative positive
        logits = self.model(x)
        logits0 = self.model(x0)
        logits1 = self.model(x1)
        a = (logits - logits0) @ (logits - logits0)
        b = (logits - logits1) @ (logits - logits1)
        temp = torch.tensor([a, b], requires_grad=True)
        self.c = nn.Softmax(dim=0)(temp)
        return self.c

    def loss_func(self):
        target = torch.tensor([0., 1.], requires_grad=True)
        self.mse_loss = self.loss_fn(self.c, target)
        return self.mse_loss

    def backward_func(self):
        self.mse_loss.backward()
        pass

    def predict(self, x):
        return self.model(x).to_list()


file = open("word_embedding_visual.txt", "r")
content = file.readlines()
node = torch.zeros(821, 2)
tot = -1
for line in content:
    tot += 1
    a, b, c = line.split(" ")
    a = float(a)
    b = float(b)
    node[tot][0] = a
    node[tot][1] = b

color = {0: "red", 1: "blue", 2: "brown", 3: "green", 4: "black", 5: "pink", 6: 'purple'}
belong = np.zeros(821)
belong[0] = 1
belong[9] = 1
belong[4] = 1
belong[6] = 1
belong[47] = 1
belong[44] = 1
belong[7] = 1
belong[11] = 1
belong0 = [0, 9, 4, 6, 47, 44, 7, 11]

belong[66] = 2
belong[391] = 2
belong1 = [66, 391]

belong[215] = 3
belong[246] = 3
belong[165] = 3
belong[307] = 3
belong[314] = 3
belong[192] = 3
belong[231] = 3
belong[232] = 3
belong[238] = 3
belong[243] = 3
belong[311] = 3
belong[244] = 3
belong[30] = 3
belong[35] = 3
belong[308] = 3
belong[79] = 3
belong[309] = 3
belong[513] = 3
belong[706] = 3
belong[707] = 3
belong[735] = 3
belong[736] = 3
belong[774] = 3
belong[770] = 3
belong[777] = 3
belong2 = [215, 246, 165, 307, 314, 192, 231, 232, 238, 243, 311, 244, 30, 35, 308, 79, 309, 513, 706, 707, 735, 736,
           774, 770, 777]

model = NeuralNetworks()
epoches = 50
learning_rate = 5e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for iter_time in range(epoches):
    for i in belong0:
        for j in belong1:
            for k in belong0:
                x = node[i]
                y = node[j]
                z = node[k]
                a = model.forward(x, y, z)
                loss = model.loss_func()
                optimizer.zero_grad()
                model.backward_func()
                optimizer.step()

    for i in belong0:
        for j in belong2:
            for k in belong0:
                x = node[i]
                y = node[j]
                z = node[k]
                a = model.forward(x, y, z)
                loss = model.loss_func()
                optimizer.zero_grad()
                model.backward_func()
                optimizer.step()

    for i in belong1:
        for j in belong0:
            for k in belong1:
                x = node[i]
                y = node[j]
                z = node[k]
                a = model.forward(x, y, z)
                loss = model.loss_func()
                optimizer.zero_grad()
                model.backward_func()
                optimizer.step()

    for i in belong1:
        for j in belong2:
            for k in belong1:
                x = node[i]
                y = node[j]
                z = node[k]
                a = model.forward(x, y, z)
                loss = model.loss_func()
                optimizer.zero_grad()
                model.backward_func()
                optimizer.step()

    for i in belong2:
        for j in belong0:
            for k in belong2:
                x = node[i]
                y = node[j]
                z = node[k]
                a = model.forward(x, y, z)
                loss = model.loss_func()
                optimizer.zero_grad()
                model.backward_func()
                optimizer.step()

    for i in belong2:
        for j in belong1:
            for k in belong2:
                x = node[i]
                y = node[j]
                z = node[k]
                a = model.forward(x, y, z)
                loss = model.loss_func()
                optimizer.zero_grad()
                model.backward_func()
                optimizer.step()
