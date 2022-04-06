import torch.nn as nn
import torch


class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduce= False)
        self.gamma = gamma
    def forward(self,prediction,target):
        log_pred = self.loss(prediction,target.squeeze())
        pred = torch.exp(-log_pred)
        loss = (1-pred).pow(self.gamma)*log_pred
        return loss.mean()