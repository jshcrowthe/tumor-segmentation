import torch.nn.functional as F
import torch

def dice_coef(logits,target,smooth= 1):
    preds = F.softmax(logits,dim=1)
    preds = preds.argmax(dim=1)
    target = target.squeeze()
    n_classes = logits.shape[1]
    results = []
    for i in range(n_classes):
        y_hat = preds == i
        y     = target == i 

        intersection = (y == y_hat).sum()

        truth = y.sum()
        count = y_hat.sum()
        results.append(2*(intersection+smooth)/(truth+count+smooth))

    return torch.stack(results).mean()
