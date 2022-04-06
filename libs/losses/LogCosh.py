import torch.nn as nn
import torch
from .Dice import DiceLoss

class LogCoshLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = DiceLoss(reduce = False)

    def forward(self,prediction,target):
        return torch.log(torch.cosh(self.loss(prediction,target))).mean()