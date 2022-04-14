import torch.nn.functional as F
import torch
import torch.nn as nn

"""
Class defining the different metrics we will use to evaluate the network 

Usage:
    metrics = Metrics()
    acc = metrics.accuracy(pred, targ)
"""

def accuracy(prediction,target):
    return (((prediction== target.squeeze())&(target.squeeze()>0)).sum(dim=(1,2,3))/(target.squeeze()>0).float().sum(dim = (1,2,3))).mean()


# Mean intersection over union

def iou(logits,target,smooth=1):
    preds = F.softmax(logits,dim=1)
    preds = preds.argmax(dim=1)
    target = target.squeeze()

    if len(logits.shape) == 5:
        target = nn.functional.one_hot(target.squeeze(),num_classes = logits.shape[1]).float().permute(0,4,1,2,3)
        preds = nn.functional.one_hot(preds.squeeze(),num_classes = logits.shape[1]).float().permute(0,4,1,2,3)
    else:
        target = nn.functional.one_hot(target.squeeze(),num_classes = logits.shape[1]).float().permute(0,3,1,2)
        preds = nn.functional.one_hot(preds.squeeze(),num_classes = logits.shape[1]).float().permute(0,3,1,2)

    preds =preds.flatten(start_dim = 1)
    y = target.flatten(start_dim = 1)

    intersection = (y*preds).sum(dim=1)

    truth = y.sum(dim=1)
    count = preds.sum(dim=1)

    return ((intersection+smooth)/(truth+count- intersection + smooth)).mean()

def dice_coef(logits,target,smooth= 1):
    sm = F.softmax(logits,dim=1)
    preds = sm.argmax(dim=1)
    target = target.squeeze()


    if len(logits.shape) == 5:
        target = nn.functional.one_hot(target.squeeze(),num_classes = logits.shape[1]).float().permute(0,4,1,2,3)
        preds = nn.functional.one_hot(preds.squeeze(),num_classes = logits.shape[1]).float().permute(0,4,1,2,3)
    else:
        target = nn.functional.one_hot(target.squeeze(),num_classes = logits.shape[1]).float().permute(0,3,1,2)
        preds = nn.functional.one_hot(preds.squeeze(),num_classes = logits.shape[1]).float().permute(0,3,1,2)

    preds =preds.flatten(start_dim = 1)
    y = target.flatten(start_dim = 1)

    intersection = (y*preds).sum(dim=1)

    truth = y.sum(dim=1)
    count = preds.sum(dim=1)

    return ((2*intersection + smooth)/(truth+count +smooth)).mean()
