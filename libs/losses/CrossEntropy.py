import torch.nn as nn

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,X,Y):
        return self.cross_entropy(X,Y.squeeze())
