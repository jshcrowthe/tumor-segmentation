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
        if len(prediction.shape) == 5:
            target = nn.functional.one_hot(target.squeeze(),num_classes = prediction.shape[1]).float().permute(0,4,1,2,3)
        else:
            target = nn.functional.one_hot(target.squeeze(),num_classes = prediction.shape[1]).float().permute(0,3,1,2)
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

class DiceLoss(nn.Module):
    def __init__(self,reduce=True,gamma=1):
        super().__init__()
        self.loss = FocalTverskyLoss(alpha = .5,gamma=gamma,reduce = reduce)
    def forward(self,predictions,target):
        return self.loss(predictions,target)

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

class JacardLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = DiceLoss(reduce = False)

    def forward(self,prediction,target):
        dice = self.loss(prediction,target)
        return (dice/(2-dice)).mean()

class LogCoshLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = DiceLoss(reduce = False)

    def forward(self,prediction,target):
        return torch.log(torch.cosh(self.loss(prediction,target))).mean()

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,prediction,target):
        return self.cross_entropy(prediction,target.squeeze())