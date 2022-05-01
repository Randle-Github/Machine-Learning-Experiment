import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
import numpy as np

from model import Model_Selector
from dataset import Datasets
from utils import train_loop, test_loop
from get_config import get_config

import matplotlib.pyplot as plt

def main(cfg):

    DatasetName = cfg["DatasetName"]
    ModleName = cfg["ModelName"]
    batch_size = cfg["batch_size"]
    learning_rate = float(cfg["learning_rate"])
    epoches = cfg["epoches"]
    LoadModel = cfg["LoadModel"]
    training_mode = cfg["training_mode"]

    training_data, test_data = Datasets(DatasetName)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = Model_Selector(ModleName).to(device)
    if LoadModel != None and os.path.exists(LoadModel+".pth"):
        model = torch.load(LoadModel+".pth")

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    if training_mode == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif training_mode == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise Exception("------No Such Training Mode------")

    loss_his = []
    for t in range(epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        temp_loss = test_loop(test_dataloader, model, loss_fn, device)
        if cfg["SaveModel"] == True:
            torch.save(model, LoadModel + ".pth")
        loss_his.append(temp_loss)
    print("Done!")

    if cfg["visual"] == True:
        plt.title("loss history")
        plt.plot(np.arange(epoches), loss_his)
        plt.show()

if __name__ == "__main__":
    cfg = get_config()
    main(cfg)