from .Tversky import FocalTverskyLoss
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self,reduce=True):
        super().__init__()
        self.loss = FocalTverskyLoss(alpha = .5,gamma=1,reduce = True)
    def forward(self,predictions,target):
        return self.loss(predictions,target)