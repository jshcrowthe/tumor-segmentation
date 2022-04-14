import torch.nn as nn
import torch

class FocalTverskyLoss(nn.Module):

    def __init__(self,reduce=True, alpha = .5 , gamma = 2,smooth = 1,weight = None):
        super().__init__()
        self.weight = weight
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
            
        preds =prediction
        y = target
        dim = 1
        if self.weight is not None:
            dim = (2,3,4)
        tp = (y*preds).sum(dim=dim)
        fn = (y*(1-preds)).sum(dim=dim)
        fp = ((1-y)*preds).sum(dim=dim)

        tversky = (tp+self.smooth)/(tp+self.alpha*fn + (1-self.alpha)*fp + self.smooth)
        
        if self.weight is not None:
            tversky = ((1-tversky).pow(self.gamma)*self.weight).sum(dim=1)
        else:
            tversky = (1-tversky).pow(self.gamma)
        if self.reduce:

            return tversky.mean()

        else:
            
            return tversky

class DiceLoss(nn.Module):
    def __init__(self,reduce=True,gamma=1,weight = None):
        super().__init__()
        self.loss = FocalTverskyLoss(alpha = .5,gamma=gamma,reduce = reduce,weight=weight)
    def forward(self,predictions,target):
        return self.loss(predictions,target)

class FocalLoss(nn.Module):

    def __init__(self, gamma=2,weight=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduce= False)
        self.loss_weighted = nn.CrossEntropyLoss(reduce= False,weight= weight)
        self.gamma = gamma
    def forward(self,prediction,target):
        log_pred = self.loss_weighted(prediction,target.squeeze())
        pred = torch.exp( -self.loss(prediction,target.squeeze()))
        loss = (1-pred).pow(self.gamma)*log_pred
        return loss.mean()

class JacardLoss(nn.Module):

    def __init__(self,weight= None):
        super().__init__()
        self.loss = DiceLoss(reduce = False,weight= weight)

    def forward(self,prediction,target):
        dice = self.loss(prediction,target)
        return (dice/(2-dice)).mean()

class LogCoshLoss(nn.Module):

    def __init__(self,weight= None):
        super().__init__()
        self.loss = DiceLoss(reduce = False,weight =weight)

    def forward(self,prediction,target):
        return torch.log(torch.cosh(self.loss(prediction,target))).mean()

class CrossEntropy(nn.Module):
    def __init__(self,weight =None):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight  = weight)

    def forward(self,prediction,target):
        return self.cross_entropy(prediction,target.squeeze())