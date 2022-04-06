import torch.nn as nn
import torch


class FocalTverskyLoss(nn.Module):

    def __init__(self,reduce=True, alpha = 1 , gamma = 2,smooth = 1):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduce = reduce
    def forward(self,prediction,target):

        prediction = prediction.softmax(dim=1)
        target = nn.functional.one_hot(target.squeeze(),num_classes = prediction.shape[1]).float().permute(0,4,1,2,3)

        preds =prediction.flatten(start_dim = 1)
        y = target.flatten(start_dim = 1)

        tp = (y*preds).sum(dim=1)
        fn = (y*(1-preds)).sum(dim=1)
        fp = ((1-y)*preds).sum(dim=1)

        tversky = (tp+self.smooth)/(tp+self.alpha*fn + (1-self.alpha)*fp + self.smooth)

        if self.reduce:

            return ((1 - tversky).pow(self.gamma)).mean()

        else:
            
            return (1 - tversky).pow(self.gamma)



