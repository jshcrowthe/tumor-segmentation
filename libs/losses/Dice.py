from .Tversky import FocalTverskyLoss
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.loss = FocalTverskyLoss(n_classes = n_classes,alpha = .5,gamma=1)
    def forward(self,predictions,target):
        return self.loss(predictions,target)