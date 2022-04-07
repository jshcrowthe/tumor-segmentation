from .Dice import DiceLoss
import torch.nn as nn

class JacardLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = DiceLoss(reduce = False)

    def forward(self,prediction,target):
        dice = self.loss(prediction,target)
        return (dice/(2-dice)).mean()
