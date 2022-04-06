import torch.nn as nn
import torch


class FocalTverskyLoss(nn.Module):

    def __init__(self,n_classes, alpha = 1 , gamma = 2,smooth = 1):
        super().__init__()

        self.n_classes = n_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self,prediction,target):

        prediction = prediction.softmax(dim=1)
        target = nn.functional.one_hot(target.squeeze(),num_classes = prediction.shape[1]).float().permute(0,4,1,2,3)

        preds =prediction.flatten(start_dim = 1)
        y = target.flatten(start_dim = 1)

        tp = (y*preds).sum()
        fn = (y*(1-preds)).sum()
        fp = ((1-y)*preds).sum()

        tversky = (tp+self.smooth)/(tp+self.alpha*fn + (1-self.alpha)*fp + self.smooth)
        return (1 - tversky).pow(self.gamma)


