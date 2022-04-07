from .Tversky import FocalTverskyLoss
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self,reduce=True,gamma=1):
        super().__init__()
        self.loss = FocalTverskyLoss(alpha = .5,gamma=gamma,reduce = reduce)
    def forward(self,predictions,target):
        return self.loss(predictions,target)
