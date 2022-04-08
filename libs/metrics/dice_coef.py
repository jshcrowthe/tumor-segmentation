import torch.nn.functional as F
import torch
import torch.nn as nn



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