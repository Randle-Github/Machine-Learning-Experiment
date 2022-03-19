import resnet_helper
import torch.nn as nn

class metricnet(nn.modules):
    def __init__(self, cfg):
        super(metricnet, self).__init__()
        self.cfg = cfg
        if self.cfg["MODEL"]["NAME"] == "resnet50":
            self.model = resnet_helper.resnet50(inplanes = 3, num_classes = cfg["MODEL"]["MAPPING_DIM"])
        elif self.cfg["MODEL"]["NAME"] == "resnet34":
             self.model = resnet_helper.resnet30(inplanes = 3, num_classes = cfg["MODEL"]["MAPPING_DIM"])
        elif self.cfg["MODEL"]["NAME"] == "resnet18":
             self.model = resnet_helper.resnet18(inplanes = 3, num_classes = cfg["MODEL"]["MAPPING_DIM"])
        elif self.cfg["MODEL"]["NAME"] == "resnet101":
             self.model = resnet_helper.resnet101(inplanes = 3, num_classes = cfg["MODEL"]["MAPPING_DIM"])
        else:
            raise Exception("no such model, the only selections are resnet18/34/50/101") 
    
    def forward(self, x):
        '''
        input_dim: (batch_num, channel, H, W)
        '''
        return self.model(x)