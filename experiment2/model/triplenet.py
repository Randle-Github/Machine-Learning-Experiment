import metricnet
import torch.nn as nn
import torch

def triple_loss(embedpos1, embedpos2, embedneg1):
    loss = torch.norm(embedneg1-embedpos1) - torch.norm(embedpos2-embedpos1)
    return loss


class triplenet(nn.modules):
    def __init__(self, cfg):
        super(triplenet, self).__init__()
        self.cfg = cfg
        self.branch_net = metricnet.metricnet(cfg)

    def forward(self, pos1, pos2, neg1):
        embedpos1 = self.branch_net(pos1)
        embedpos2 = self.branch_net(pos2)
        embedneg1 = self.branch_net(neg1)
        return embedpos1, embedpos2, embedneg1
