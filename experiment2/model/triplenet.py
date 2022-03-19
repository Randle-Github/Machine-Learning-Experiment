import resnet
import torch.nn as nn

class triplenet(nn.modules):
    def __init__(self, cfg):
        super(triplenet, self).__init__()
        if cfg["MODEL"]["NAME"] == "resnet50":
            self.metric = 
        elif 
