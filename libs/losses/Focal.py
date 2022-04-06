import torch.nn as nn
import torch


class FocalLoss(nn.Module):

    def __init__(self,n_classes, gamma=2):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduce= False)
        self.n_classes = n_classes
        self.gamma = gamma
    def forward(self,prediction,target):
        log_pred = self.loss(prediction,target.squeeze())
        pred = torch.exp(-log_pred)
        loss = (1-pred).pow(self.gamma)*log_pred
        return loss.mean()